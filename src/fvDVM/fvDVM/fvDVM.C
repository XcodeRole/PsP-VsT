/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2013 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include <mpi.h>
#include "fvDVM.H"
#include "constants.H"
#include "fvm.H"
#include "calculatedMaxwellFvPatchField.H"
#include "symmetryModFvPatchField.H"
#include "pressureInFvPatchField.H"
#include "pressureOutFvPatchField.H"
#include "scalarIOList.H"
#include "fieldMPIreducer.H"
#include <omp.h>
#include "tickTock.H"
#include <stdlib.h>
#include "PstreamGlobals.H"
#include "processorFvPatch.H"
#define NUM_THREADS 1

using namespace Foam::constant;
using namespace Foam::constant::mathematical;

#if FOAM_MAJOR <= 3
    #define BOUNDARY_FIELD_REF boundaryField()
#else
    #define BOUNDARY_FIELD_REF boundaryFieldRef()
#endif


// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(fvDVM, 0);
}

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

void Foam::fvDVM::setDVgrid
(
    scalarField& weights,
    scalarField& Xis,
    scalar xiMin,
    scalar xiMax,
    label nXi
)
{
    // Read from file ./constant/Xis and ./constant/weights
    scalarIOList xiList
    (
        IOobject
        (
             "Xis",
             time_.caseConstant(),
             mesh_,
             IOobject::MUST_READ,
             IOobject::NO_WRITE
        )
    );

    scalarIOList weightList
    (
        IOobject
        (
             "weights",
             time_.caseConstant(),
             mesh_,
             IOobject::MUST_READ,
             IOobject::NO_WRITE
        )
    );

    // Check
    //if (
            //xiList.size() != nXi 
         //|| weightList.size() != nXi
       //)
    //{
        //WarningIn("setDVgrid")
            //<< "Num of discreteVelocity not consistent" << endl;
        //std::exit(-1);
    //}


    //if ( 
           //xiList[0]  != xiMin
        //|| xiList[nXi-1] != xiMax)
    //{
        //WarningIn("setDVgrid")
            //<< "xi not consistant" <<endl;
        //std::exit(-1);
    //}


    for (label i = 0; i < nXi ; i++)
    {
        weights[i] = weightList[i];
        Xis[i] = xiList[i];
    }
}

void Foam::fvDVM::initialiseDV()
{
    scalarField weights1D(nXiPerDim_);
    scalarField Xis(nXiPerDim_);
#ifdef VMESH
    xiMax_.value()=0.0;
//we have polyMesh and vMesh at the beginning
	string mesh_name = args_.path() / word("constant") / word("polyMesh");//polyMesh dir
	string vmesh_name = args_.path() / word("constant") / word("vMesh");//vMesh dir
	string temp_name = args_.path() / word("constant") / word("tMesh");//temp name

	//1.change folder name, polyMesh-->tMesh,vMesh-->polyMesh
	//as OF only recognizes certain path in the couse of reading mesh files
	
	if(mpiReducer_.rank() == 0 || args_.optionFound("parallel")) {
          std::rename(mesh_name.c_str(), temp_name.c_str());
          std::rename(vmesh_name.c_str(), mesh_name.c_str());
       }

	//2.read and construct vmesh 
	Foam::fvMesh vmesh
	(
		Foam::IOobject
		(
			Foam::fvMesh::defaultRegion,
			time_.timeName(),
			time_,
			Foam::IOobject::MUST_READ
		)
	);


	//3.change the folder name back, polyMesh-->vMesh,tMesh-->polyMesh,
	//so no one would know we change the name secretly :D
        if(mpiReducer_.rank() == 0 || args_.optionFound("parallel")) {
            std::rename(mesh_name.c_str(), vmesh_name.c_str());
            std::rename(temp_name.c_str(), mesh_name.c_str());
       }

	labelField  symmXtgID;
	labelField  symmYtgID;
	labelField  symmZtgID;

	if (mesh_.nSolutionD() == 3)    //3D(X & Y & Z)
	{
		nXiX_ = nXiY_ = nXiZ_ = vmesh.C().size();
		nXi_ = vmesh.C().size();

		weightsGlobal.setSize(nXi_);
		XisGlobal.setSize(nXi_);
		symmXtgID.setSize(nXi_);
		symmYtgID.setSize(nXi_);
		symmZtgID.setSize(nXi_);

		label i;
		for (i = 0; i < nXi_; i++) {
			vector xi(vmesh.C()[i].x(), vmesh.C()[i].y(), vmesh.C()[i].z());
			scalar weight(vmesh.V()[i]);
			weightsGlobal[i] = weight;
			XisGlobal[i] = xi;
			xiMax_.value()=max(xiMax_.value(),mag(xi));
		}
		/*

				for (label iz = 0; iz < nXiZ_; iz++)
				{
					for (label iy = 0; iy < nXiY_; iy++)
					{
						for (label ix = 0; ix < nXiZ_; ix++)
						{
							scalar weight = weights1D[iz]*weights1D[iy]*weights1D[ix];
							vector xi(Xis[ix], Xis[iy], Xis[iz]);
							weightsGlobal[i] = weight;
							XisGlobal[i] = xi;
							symmXtgID[i] = iz*nXiY_*nXiX_ + iy*nXiX_ + (nXiX_ - ix -1);
							symmYtgID[i] = iz*nXiY_*nXiX_ + (nXiY_ - iy - 1)*nXiX_ + ix;
							symmZtgID[i] = (nXiZ_ - iz -1)*nXiY_*nXiX_ + iy*nXiX_ + ix;
							i++;
						}
					}
				}
		*/
	}
	else
	{
		if (mesh_.nSolutionD() == 2)    //2D (X & Y)
		{
			nXiX_ = nXiY_ = vmesh.C().size();
			nXiZ_ = 1;
			nXi_ = vmesh.C().size();
			weightsGlobal.setSize(nXi_);
			XisGlobal.setSize(nXi_);
			symmXtgID.setSize(nXi_);
			symmYtgID.setSize(nXi_);
			symmZtgID.setSize(nXi_);
			label i;
			for (i = 0; i < nXi_; i++) {
				vector xi(vmesh.C()[i].x(), vmesh.C()[i].y(), 0.0);
				scalar weight(vmesh.V()[i]);
				weightsGlobal[i] = weight;
				XisGlobal[i] = xi;
				xiMax_.value()=max(xiMax_.value(),mag(xi));
			}
			/*
						for (label iy = 0; iy < nXiY_; iy++)
						{
							for (label ix = 0; ix < nXiX_; ix++)
							{
								scalar weight = weights1D[iy]*weights1D[ix]*1;
								vector xi(Xis[ix], Xis[iy], 0.0);
								weightsGlobal[i] = weight;
								XisGlobal[i] = xi;
								symmXtgID[i] = iy*nXiX_ + (nXiX_ - ix -1);
								symmYtgID[i] = (nXiY_ - iy - 1)*nXiX_ + ix;
								symmZtgID[i] = 0;
								i++;
							}
						}
			*/
		}
		else    //1D (X)
		{
			nXiX_ = vmesh.C().size();
			nXiY_ = nXiZ_ = 1;
			nXi_ = vmesh.C().size();
			weightsGlobal.setSize(nXi_);
			XisGlobal.setSize(nXi_);
			symmXtgID.setSize(nXi_);
			symmYtgID.setSize(nXi_);
			symmZtgID.setSize(nXi_);
			label i;
			for (i = 0; i < nXi_; i++) {
				vector xi(vmesh.C()[i].x(), 0.0, 0.0);
				scalar weight(vmesh.V()[i]);
				weightsGlobal[i] = weight;
				XisGlobal[i] = xi;
				xiMax_.value()=max(xiMax_.value(),mag(xi));
			}
			/*
						for (label ix = 0; ix < nXiX_; ix++)
						{
							scalar weight = weights1D[ix]*1*1;
							vector xi(Xis[ix], 0.0, 0.0);
							weightsGlobal[i] = weight;
							XisGlobal[i] = xi;
							symmXtgID[i] = (nXiX_ - ix -1);
							symmYtgID[i] = 0;
							symmZtgID[i] = 0;
							i++;
						}
			*/
		}
	}
#else
    //get discrete velocity points and weights
    setDVgrid
    (
         weights1D,
         Xis, 
         xiMin_.value(), 
         xiMax_.value(), 
         nXiPerDim_
    );

    labelField  symmXtgID;
    labelField  symmYtgID;
    labelField  symmZtgID;

    if (mesh_.nSolutionD() == 3)    //3D(X & Y & Z)
    {
        nXiX_ = nXiY_ = nXiZ_ = nXiPerDim_;
        nXi_ = nXiX_*nXiY_*nXiZ_;

        weightsGlobal.setSize(nXi_);
        XisGlobal.setSize(nXi_);
        symmXtgID.setSize(nXi_);
        symmYtgID.setSize(nXi_);
        symmZtgID.setSize(nXi_);

        label i = 0;
        for (label iz = 0; iz < nXiZ_; iz++)
        {
            for (label iy = 0; iy < nXiY_; iy++)
            {
                for (label ix = 0; ix < nXiZ_; ix++)
                {
                    scalar weight = weights1D[iz]*weights1D[iy]*weights1D[ix];
                    vector xi(Xis[ix], Xis[iy], Xis[iz]);
                    weightsGlobal[i] = weight;
                    XisGlobal[i] = xi;
                    symmXtgID[i] = iz*nXiY_*nXiX_ + iy*nXiX_ + (nXiX_ - ix -1);
                    symmYtgID[i] = iz*nXiY_*nXiX_ + (nXiY_ - iy - 1)*nXiX_ + ix;
                    symmZtgID[i] = (nXiZ_ - iz -1)*nXiY_*nXiX_ + iy*nXiX_ + ix;
                    i++;
                }
            }
        }
    }
    else
    {
        if (mesh_.nSolutionD() == 2)    //2D (X & Y)
        {
            nXiX_ = nXiY_ = nXiPerDim_;
            nXiZ_ = 1;
            nXi_ = nXiX_*nXiY_*nXiZ_;
            weightsGlobal.setSize(nXi_);
            XisGlobal.setSize(nXi_);
            symmXtgID.setSize(nXi_);
            symmYtgID.setSize(nXi_);
            symmZtgID.setSize(nXi_);
            label i = 0;
            for (label iy = 0; iy < nXiY_; iy++)
            {
                for (label ix = 0; ix < nXiX_; ix++)
                {
                    scalar weight = weights1D[iy]*weights1D[ix]*1;
                    vector xi(Xis[ix], Xis[iy], 0.0);
                    weightsGlobal[i] = weight;
                    XisGlobal[i] = xi;
                    symmXtgID[i] = iy*nXiX_ + (nXiX_ - ix -1);
                    symmYtgID[i] = (nXiY_ - iy - 1)*nXiX_ + ix;
                    symmZtgID[i] = 0;
                    i++;
                }
            }
        }
        else    //1D (X)
        {
            nXiX_ = nXiPerDim_;
            nXiY_ = nXiZ_ = 1;
            nXi_ = nXiX_*nXiY_*nXiZ_;
            weightsGlobal.setSize(nXi_);
            XisGlobal.setSize(nXi_);
            symmXtgID.setSize(nXi_);
            symmYtgID.setSize(nXi_);
            symmZtgID.setSize(nXi_);
            label i = 0;
            for (label ix = 0; ix < nXiX_; ix++)
            {
                scalar weight = weights1D[ix]*1*1;
                vector xi(Xis[ix], 0.0, 0.0);
                weightsGlobal[i] = weight;
                XisGlobal[i] = xi;
                symmXtgID[i] = (nXiX_ - ix -1);
                symmYtgID[i] = 0;              
                symmZtgID[i] = 0;              
                i++;
            }
        }
    }
#endif
    if(mpiReducer_.rank() == 0)
    {
        Info<< "fvDVM : Allocated " << XisGlobal.size()
            << " discrete velocities" << endl;
    }
    label nA = nXi_ / mpiReducer_.nproc();
    label nB = nXi_ - nA*mpiReducer_.nproc();
    label nXiPart = nA + (label)(mpiReducer_.rank() < nB);
    DV_.setSize(nXiPart);
    
    dvSize = nXiPart;
    cellSize = mesh_.nCells();
    faceSize = mesh_.nInternalFaces();

    if(mpiReducer_.rank() == 0)
    {
        Info << "nproc    " << mpiReducer_.nproc()  << endl;
        Info << "nXisPart " << nXiPart << endl;
    }

    // _gTildeVol = new scalar[dvSize*cellSize];
    // _hTildeVol = new scalar[dvSize*cellSize];
    // _gBarPvol  = new scalar[dvSize*cellSize];
    // _hBarPvol  = new scalar[dvSize*cellSize];
    // _gSurf     = new scalar[dvSize*faceSize];
    // _hSurf     = new scalar[dvSize*faceSize];
    Pout<<"dvSize*cellSize:"<<dvSize*cellSize<<endl;
    _gTildeVol = (scalar*)malloc(dvSize*cellSize*sizeof(scalar));
    _hTildeVol = (scalar*)malloc(dvSize*cellSize*sizeof(scalar));
    _gBarPvol  = (scalar*)malloc(dvSize*cellSize*sizeof(scalar));
    _hBarPvol  = (scalar*)malloc(dvSize*cellSize*sizeof(scalar));
    _gSurf     = (scalar*)malloc(dvSize*faceSize*sizeof(scalar));
    _hSurf     = (scalar*)malloc(dvSize*faceSize*sizeof(scalar));
    scalar* rho = rhoVol_.data();
    vector* U = Uvol_.data();
    scalar* T = Tvol_.data();
    scalar R = R_.value();
    label D = mesh_.nSolutionD();
    #pragma omp parallel for num_threads(NUM_THREADS)
    for(size_t i=0;i<dvSize;i++){
        for(size_t internal_id=0;internal_id<cellSize;internal_id++){

            scalar cSqrByRT = magSqr(U[internal_id] - XisGlobal[i])/(R*T[internal_id]);

            scalar gEqBGK = rho[internal_id]/pow(sqrt(2.0*pi*R*T[internal_id]),D)*exp(-cSqrByRT/2.0);

            _gTildeVol[i*cellSize+internal_id] = gEqBGK;
            _hTildeVol[i*cellSize+internal_id] =  (KInner_ + 3.0 - D)  *gEqBGK*R*T[internal_id];

        }
    }
    for(size_t i=0;i<dvSize;i++){
        volScalarField* gTildeVol_ = new volScalarField
        (
            IOobject
            (
                "gTildeVol" + Foam::name(i),
                mesh_.time().timeName(),
                mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            mesh_,
            dimMass*pow3(dimTime/dimLength)/pow3(dimLength),
            _gTildeVol+cellSize*i
        );
        volScalarField* hTildeVol_ = new volScalarField
        (
            IOobject
            (
                "hTildeVol" + Foam::name(i),
                mesh_.time().timeName(),
                mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            mesh_,
            dimMass*(dimTime/dimLength)/pow3(dimLength),
            _hTildeVol+cellSize*i
        );
        volScalarField* gBarPvol_ =new  volScalarField
        (
            IOobject
            (
                "gBarPvol" + Foam::name(i),
                mesh_.time().timeName(),
                mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            mesh_,
            dimMass*pow3(dimTime/dimLength)/pow3(dimLength),
            _gBarPvol+cellSize*i,
            "fixedGradient"
        );
        volScalarField* hBarPvol_ = new volScalarField
        (
            IOobject
            (
                "hBarPvol" + Foam::name(i),
                mesh_.time().timeName(),
                mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            mesh_,
            dimMass*(dimTime/dimLength)/pow3(dimLength),
            _hBarPvol+cellSize*i,
            "fixedGradient"
        );
        surfaceScalarField* gSurf_ =new surfaceScalarField
        (
            IOobject
            (
                "gSurf" + Foam::name(i),
                mesh_.time().timeName(),
                mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            mesh_,
            dimMass*pow3(dimTime/dimLength)/pow3(dimLength),
            _gSurf+faceSize*i
        );
        surfaceScalarField*  hSurf_ =new surfaceScalarField
        (
            IOobject
            (
                "hSurf" + Foam::name(i),
                mesh_.time().timeName(),
                mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            mesh_,
            dimMass*(dimTime/dimLength)/pow3(dimLength),
            _hSurf+faceSize*i
        );
        gTildeVol.push_back(gTildeVol_);
        hTildeVol.push_back(hTildeVol_);
        gBarPvol.push_back(gBarPvol_);
        hBarPvol.push_back(hBarPvol_);
        gSurf.push_back(gSurf_);
        hSurf.push_back(hSurf_);
        DV_.set
        (
            i,
            new discreteVelocity
            (
                *this,
                mesh_,
                time_,
                weightsGlobal[i],
                dimensionedVector( "xi", dimLength/dimTime, XisGlobal[i]),
                i,
                symmXtgID[i],
                symmYtgID[i],
                symmZtgID[i],
                *gTildeVol_,
                *hTildeVol_,
                *gBarPvol_,
                *hBarPvol_,
                *gSurf_,
                *hSurf_
            )
        );
    }

    //为sendbuf以及recvbuf申请空间
    forAll(mesh_.boundary(),patchi){
        if( mesh_.boundary()[patchi].type() == "processor" ){
            size_t bufSize = mesh_.boundary()[patchi].size()*dvSize*2;//乘2是因为phiBar包括gbar和hbar
            scalar* SendBuf_tmp = (scalar*)malloc(bufSize*sizeof(scalar));
            scalar* RecvBuf_tmp = (scalar*)malloc(bufSize*sizeof(scalar));
            sendBuf[patchi] = SendBuf_tmp;
            recvBuf[patchi] = RecvBuf_tmp;
        }
    }
    // label chunk = 0;
    // label gid = 0;

    // forAll(DV_, i)
    // {
    //     DV_.set
    //     (
    //         i,
    //         new discreteVelocity
    //         (
    //             *this,
    //             mesh_,
    //             time_,
    //             weightsGlobal[i],
    //             dimensionedVector( "xi", dimLength/dimTime, XisGlobal[i]),
    //             i,
    //             symmXtgID[i],
    //             symmYtgID[i],
    //             symmZtgID[i]
    //         )
    //     );
    // }
}

void Foam::fvDVM::setCalculatedMaxwellRhoBC()
{
#if FOAM_MAJOR <= 3
    GeometricField<scalar, fvPatchField, volMesh>::GeometricBoundaryField& 
        rhoBCs = rhoVol_.boundaryField();
#else
    GeometricField<scalar, fvPatchField, volMesh>::Boundary& 
        rhoBCs = rhoVol_.BOUNDARY_FIELD_REF;
#endif
    forAll(rhoBCs, patchi)
    {
        if (rhoBCs[patchi].type() == "calculatedMaxwell")
        {

            const vectorField& SfPatch = mesh_.Sf().boundaryField()[patchi];
            calculatedMaxwellFvPatchField<scalar>& rhoPatch = 
                refCast<calculatedMaxwellFvPatchField<scalar> >(rhoBCs[patchi]);
            const vectorField& Upatch = Uvol_.boundaryField()[patchi];
            const scalarField& Tpatch = Tvol_.boundaryField()[patchi];

            forAll(rhoPatch, facei)
            {
                vector faceSf = SfPatch[facei];
                rhoPatch.inComingByRho()[facei] = 0; // set to zero
                forAll(DV_, dvi) // add one by one
                {
                    vector xi = DV_[dvi].xi().value();
                    scalar weight = DV_[dvi].weight();
                    if ( (xi & faceSf) < 0) //inComing
                    {
                        rhoPatch.inComingByRho()[facei] += 
                          - weight*(xi & faceSf)
                          *DV_[dvi].equilibriumMaxwellByRho
                          (
                              Upatch[facei], 
                              Tpatch[facei]
                          );
                    }
                }
            }

            if(args_.optionFound("dvParallel"))
                mpiReducer_.reduceField(rhoPatch.inComingByRho());
        }

    }
}

void Foam::fvDVM::setSymmetryModRhoBC()
{
    //prepare the container (set size) to store all DF on the patchi
#if FOAM_MAJOR <= 3
    GeometricField<scalar, fvPatchField, volMesh>::GeometricBoundaryField& 
        rhoBCs = rhoVol_.boundaryField();
#else
    GeometricField<scalar, fvPatchField, volMesh>::Boundary& 
        rhoBCs = rhoVol_.BOUNDARY_FIELD_REF;
#endif
    forAll(rhoBCs, patchi)
    {
        label ps = rhoBCs[patchi].size();
        if (rhoBCs[patchi].type() == "symmetryMod")
        {
            symmetryModFvPatchField<scalar>& rhoPatch = 
                refCast<symmetryModFvPatchField<scalar> >(rhoBCs[patchi]);
            rhoPatch.dfContainer().setSize(ps*nXi_*2); //*2 means g and h
        }
    }
}

void Foam::fvDVM::updateGHbarPvol()
{
    scalar dt = time_.deltaTValue();
    scalar* rho = rhoVol_.data();
    vector* U = Uvol_.data();
    scalar* T = Tvol_.data();
    vector* q = qVol_.data();
    scalar* tau = tauVol_.data();
    scalar R = R_.value();
    label D = mesh_.nSolutionD();

    #pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS)
    for(size_t dvi=0;dvi<dvSize;dvi++){
        // scalar* __restrict   gBarPvol_p = gBarPvol[dvi]->data();
        // scalar* __restrict   hBarPvol_p = hBarPvol[dvi]->data();
        // scalar* __restrict   gTildeVol_p = gTildeVol[dvi]->data();
        // scalar* __restrict   hTildeVol_p = hTildeVol[dvi]->data();
        #pragma GCC unroll (16)
        for(size_t internal_id = 0;internal_id<cellSize;internal_id++){
            // if constexpr (vol){
            scalar relaxFactor = 1.5*dt/(2.0*tau[internal_id] + dt);
            // }else{
            //     relaxFactor = 0.5*dt/(2.0*tau[internal_id] + 0.5*dt);
            // }

            scalar cSqrByRT = magSqr(U[internal_id]-XisGlobal[dvi])/(R*T[internal_id]);
            
            scalar cqBy5pRT = ((XisGlobal[dvi] - U[internal_id])& q[internal_id])/(5.0*rho[internal_id]*R*T[internal_id]*R*T[internal_id]);    
            scalar gEqBGK = rho[internal_id]/pow(sqrt(2.0*pi*R*T[internal_id]),D)*exp(-cSqrByRT/2.0);
                
            _gBarPvol[dvi*cellSize+internal_id] = (1.0 - relaxFactor)*_gTildeVol[dvi*cellSize+internal_id] + relaxFactor*( 1.0 + (1.0 - Pr_)*cqBy5pRT*(cSqrByRT - D - 2.0) )*gEqBGK;
            _hBarPvol[dvi*cellSize+internal_id] = (1.0 - relaxFactor)*_hTildeVol[dvi*cellSize+internal_id] + relaxFactor*( (KInner_ + 3.0 - D) + (1.0 - Pr_)*cqBy5pRT*((cSqrByRT - D)*(KInner_ + 3.0 - D) - 2*KInner_) )*gEqBGK*R*T[internal_id];
            
        }
    }
}

//合并所有dv之间的点对点通信
void Foam::fvDVM::GHbarPvolComm()
{
    label nReq = Pstream::nRequests();
    forAll(mesh_.boundary(),patchi){
        if(mesh_.boundary()[patchi].type() == "processor"){
            //1. 获取邻居进程号
            int neighbProcNo = refCast<const processorFvPatch>(mesh_.boundary()[patchi]).neighbProcNo();

            //2. fill ghbar
            const label* faceCell = mesh_.boundary()[patchi].faceCells().cdata();
            label patchSize = mesh_.boundary()[patchi].size();
            for(int dvi=0;dvi<dvSize; dvi++){
                discreteVelocity& dv = DV_[dvi];
                for(int facei=0;facei<patchSize;facei++){
                    sendBuf[patchi][patchSize*2*dvi+facei] = dv.gBarPvol()[faceCell[facei]]; //fill gbar
                    sendBuf[patchi][patchSize*2*dvi+patchSize+facei] = dv.hBarPvol()[faceCell[facei]]; //fill hbar
                }
            }

            //3. 非阻塞通信
            MPI_Request SendRequest;
            MPI_Isend(
                sendBuf[patchi], 
                mesh_.boundary()[patchi].size()*dvSize*2,
                MPI_DOUBLE, 
                neighbProcNo,
                0, 
                PstreamGlobals::MPICommunicators_[0],
                &SendRequest
            );
            PstreamGlobals::outstandingRequests_.append(SendRequest);

            MPI_Request RecvRequest;
            MPI_Irecv(
                recvBuf[patchi], 
                mesh_.boundary()[patchi].size()*dvSize*2,
                MPI_DOUBLE, 
                neighbProcNo,
                0, 
                PstreamGlobals::MPICommunicators_[0],
                &RecvRequest
            );
            PstreamGlobals::outstandingRequests_.append(RecvRequest);
        }
    }
    //waitall
    Pstream::waitRequests(nReq);

    //4. memory copy from recv buf to processorboundary
    forAll(mesh_.boundary(),patchi){
        if( mesh_.boundary()[patchi].type() == "processor" ){
            label patchSize = mesh_.boundary()[patchi].size();
            for(int dvi=0;dvi<dvSize; dvi++){
                discreteVelocity& dv = DV_[dvi];
                memcpy(DV_[dvi].gBarPvol().BOUNDARY_FIELD_REF[patchi].data(),recvBuf[patchi]+patchSize*2*dvi,patchSize*sizeof(scalar));
                memcpy(DV_[dvi].hBarPvol().BOUNDARY_FIELD_REF[patchi].data(),recvBuf[patchi]+patchSize*2*dvi+patchSize,patchSize*sizeof(scalar));
            }
        }
    }

    forAll(DV_, DVid)
        DV_[DVid].GHbarPvolComm();
}
void Foam::fvDVM::updateGrad()
{
    forAll(DV_, DVid)
        DV_[DVid].updateGrad();
}

void Foam::fvDVM::updateGHbarSurf()
{
    const labelUList& owner = mesh_.owner();
    const labelUList& neighbour = mesh_.neighbour();
    const surfaceScalarField& magSf = mesh_.magSf();
    const vectorField& Cf = mesh_.Cf();
    const surfaceVectorField& Sf = mesh_.Sf();
    const vectorField& C = mesh_.C();
    scalar dt = time_.deltaTValue();
    #pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS)
    for(size_t dvi=0;dvi<dvSize;dvi++){
        discreteVelocity& dv = DV_[dvi];
        volVectorField& gBarPgrad_ = dv.gBarPgrad();
        volVectorField& hBarPgrad_ = dv.hBarPgrad();
        volScalarField& gBarPvol_ = dv.gBarPvol();
        volScalarField& hBarPvol_ = dv.hBarPvol();
        surfaceScalarField& gSurf_ = dv.gSurf();
        surfaceScalarField& hSurf_ = dv.hSurf();
        // #pragma omp critical
        // {
            forAll(gBarPgrad_.BOUNDARY_FIELD_REF, patchi)
            {
                const vectorField n
                (
                    Sf.boundaryField()[patchi]
                /magSf.boundaryField()[patchi]
                );

                if ( 
                    gBarPvol_.BOUNDARY_FIELD_REF[patchi].type() != "empty" 
                    && gBarPvol_.BOUNDARY_FIELD_REF[patchi].type() != "processor"
                    && gBarPvol_.BOUNDARY_FIELD_REF[patchi].type() != "symmetryPlane"
                    && gBarPvol_.BOUNDARY_FIELD_REF[patchi].type() != "DVMsymmetry"
                    && gBarPvol_.BOUNDARY_FIELD_REF[patchi].type() != "cyclic"
                    && gBarPvol_.BOUNDARY_FIELD_REF[patchi].type() != "processorCyclic"
                ) // only for fixed gradient g/hBarPvol
                {
                    // normal component of the grad field
                    fixedGradientFvPatchField<scalar>& gBarPvolPatch = 
                        refCast<fixedGradientFvPatchField<scalar> >
                        (gBarPvol_.BOUNDARY_FIELD_REF[patchi]);

                    fixedGradientFvPatchField<scalar>& hBarPvolPatch = 
                        refCast<fixedGradientFvPatchField<scalar> >
                        (hBarPvol_.BOUNDARY_FIELD_REF[patchi]);

                    forAll(gBarPvolPatch, pFacei)
                    {
                        gBarPvolPatch.gradient()[pFacei] =
                            gBarPgrad_.BOUNDARY_FIELD_REF[patchi][pFacei]&n[pFacei];
                        hBarPvolPatch.gradient()[pFacei] =
                            hBarPgrad_.BOUNDARY_FIELD_REF[patchi][pFacei]&n[pFacei];
                    }
                }
            }
        // }


        const Field<scalar>& iGbarPvol = gBarPvol_;
        const Field<scalar>& iHbarPvol = hBarPvol_;
        const Field<vector>& iGbarPgrad = gBarPgrad_;
        const Field<vector>& iHbarPgrad = hBarPgrad_;

        // This is what we want to update in this function
        Field<scalar>& iGsurf = gSurf_;
        Field<scalar>& iHsurf = hSurf_;

        vector xii = XisGlobal[dvi];

        // internal faces first
        forAll(owner, facei)
        {
            label own = owner[facei];
            label nei = neighbour[facei];
            if ((xii&Sf[facei]) >=  VSMALL) // comming from own
            {

                iGsurf[facei] = iGbarPvol[own] 
                + (iGbarPgrad[own]&(Cf[facei] - C[own] - 0.5*xii*dt));

                iHsurf[facei] = iHbarPvol[own]
                + (iHbarPgrad[own]&(Cf[facei] - C[own] - 0.5*xii*dt));

            }
            // Debug, no = 0, =0 put to > 0
            else if ((xii&Sf[facei]) < -VSMALL) // comming form nei
            {
                iGsurf[facei] = iGbarPvol[nei]
                + (iGbarPgrad[nei]&(Cf[facei] - C[nei] - 0.5*xii*dt));
                iHsurf[facei] = iHbarPvol[nei]
                + (iHbarPgrad[nei]&(Cf[facei] - C[nei] - 0.5*xii*dt));
            }
            else 
            {
                iGsurf[facei] = 0.5*
                (
                    iGbarPvol[nei] + ((iGbarPgrad[nei])
                &(Cf[facei] - C[nei] - 0.5*xii*dt))
                + iGbarPvol[own] + ((iGbarPgrad[own])
                &(Cf[facei] - C[own] - 0.5*xii*dt))
                );
                iHsurf[facei] = 0.5*
                (
                iHbarPvol[nei] + ((iHbarPgrad[nei])
                &(Cf[facei] - C[nei] - 0.5*xii*dt))
                + iHbarPvol[own] + ((iHbarPgrad[own])
                &(Cf[facei] - C[own] - 0.5*xii*dt))
                );
            }
        }

        // boundary faces
        #pragma omp critical
        {
            forAll(gSurf_.BOUNDARY_FIELD_REF, patchi)
            {
                word type = gSurf_.BOUNDARY_FIELD_REF[patchi].type();
                fvsPatchField<scalar>& gSurfPatch = gSurf_.BOUNDARY_FIELD_REF[patchi];
                fvsPatchField<scalar>& hSurfPatch = hSurf_.BOUNDARY_FIELD_REF[patchi];
                const fvsPatchField<vector>& SfPatch =
                    mesh_.Sf().boundaryField()[patchi];
                const fvsPatchField<vector>& CfPatch =
                    mesh_.Cf().boundaryField()[patchi];
                const labelUList& faceCells = mesh_.boundary()[patchi].faceCells();

                const fvPatchScalarField& rhoVolPatch = 
                    rhoVol_.boundaryField()[patchi];
                const fvPatchScalarField& TvolPatch = 
                    Tvol_.boundaryField()[patchi];
                const labelUList& pOwner = mesh_.boundary()[patchi].faceCells();
                
                //- NOTE: outging DF can be treate unifily for all BCs, including processor BC
                if (type == "zeroGradient")
                {
                    gSurfPatch == gBarPvol_.BOUNDARY_FIELD_REF[patchi].patchInternalField();
                    hSurfPatch == hBarPvol_.BOUNDARY_FIELD_REF[patchi].patchInternalField();
                }
                else if (type == "mixed")
                {
                    //check each boundary face in the patch
                    forAll(gSurfPatch, facei)
                    {
                        //out or in ?
                        if ((xii&SfPatch[facei]) > 0 ) // outgoing
                        {
                            gSurfPatch[facei] = iGbarPvol[faceCells[facei]] 
                            + ((iGbarPgrad[faceCells[facei]])
                            &(CfPatch[facei] - C[faceCells[facei]] - 0.5*xii*dt));
                            hSurfPatch[facei] = iHbarPvol[faceCells[facei]] 
                            + ((iHbarPgrad[faceCells[facei]])
                            &(CfPatch[facei] - C[faceCells[facei]] - 0.5*xii*dt));
                        //incoming and parallel to face, not changed.
                        }
                    }
                }
                else if (type == "farField")
                {
                    //check each boundary face in the patch
                    forAll(gSurfPatch, facei)
                    {
                        //out or in ?
                        if ((xii&SfPatch[facei]) > 0 ) // outgoing
                        {
                            gSurfPatch[facei] = iGbarPvol[faceCells[facei]] 
                            + ((iGbarPgrad[faceCells[facei]])
                            &(CfPatch[facei] - C[faceCells[facei]] - 0.5*xii*dt));
                            hSurfPatch[facei] = iHbarPvol[faceCells[facei]] 
                            + ((iHbarPgrad[faceCells[facei]])
                            &(CfPatch[facei] - C[faceCells[facei]] - 0.5*xii*dt));
                        //incoming and parallel to face, not changed.
                        }
                        else // incomming, set to be equlibrium, give rho and T, extropolate U
                        {
                            // set to maxwellian
                            gSurfPatch[facei] = rhoVolPatch[facei]
                            *dv.equilibriumMaxwellByRho
                                (
                                    Uvol_[pOwner[facei]],
                                    TvolPatch[facei]
                                );
                            hSurfPatch[facei] = 
                                gSurfPatch[facei]*(R_.value()*TvolPatch[facei])
                            *(KInner_ + 3 - mesh_.nSolutionD());
                        }
                    }
                }
                else if (type == "maxwellWall")
                {
                    calculatedMaxwellFvPatchField<scalar>& rhoPatch = 
                        refCast<calculatedMaxwellFvPatchField<scalar> >
                        (rhoVol_.BOUNDARY_FIELD_REF[patchi]); //DEBUG

                    forAll(gSurfPatch, facei)
                    {
                        vector faceSf= SfPatch[facei];
                        if ((xii&faceSf) >  0 ) // outgoing
                        {
                            gSurfPatch[facei] = iGbarPvol[faceCells[facei]] 
                            + ((iGbarPgrad[faceCells[facei]])
                            &(CfPatch[facei] - C[faceCells[facei]] - 0.5*xii*dt));
                            hSurfPatch[facei] = iHbarPvol[faceCells[facei]] 
                            + ((iHbarPgrad[faceCells[facei]])
                            &(CfPatch[facei] - C[faceCells[facei]] - 0.5*xii*dt));

                            rhoPatch.outGoing()[facei] += //add outgoing normal momentum flux to outGoing container
                                weightsGlobal[dvi]*(xii&faceSf)*gSurfPatch[facei];
                        }
                    }
                }
                else if (type == "processor"
                    || type == "cyclic"
                    || type == "processorCyclic"
                    ) // parallel
                {
                    forAll(gSurfPatch, facei)
                    {
                        vector faceSf= SfPatch[facei];
                        if ((xii&faceSf) >  VSMALL ) // outgoing
                        {
                            gSurfPatch[facei] = iGbarPvol[faceCells[facei]] 
                            + ((iGbarPgrad[faceCells[facei]])
                            &(CfPatch[facei] - C[faceCells[facei]] - 0.5*xii*dt));
                            hSurfPatch[facei] = iHbarPvol[faceCells[facei]] 
                            + ((iHbarPgrad[faceCells[facei]])
                            &(CfPatch[facei] - C[faceCells[facei]] - 0.5*xii*dt));
                        } 
                        else if ((xii&faceSf) <  -VSMALL )//incomming from processor boundaryField
                        {
                            gSurfPatch[facei] = gBarPvol_.BOUNDARY_FIELD_REF[patchi][facei]
                            + ((gBarPgrad_.BOUNDARY_FIELD_REF[patchi][facei])
                            &(CfPatch[facei] - mesh_.C().boundaryField()[patchi][facei] - 0.5*xii*dt));
                            hSurfPatch[facei] = hBarPvol_.BOUNDARY_FIELD_REF[patchi][facei]
                            + ((hBarPgrad_.BOUNDARY_FIELD_REF[patchi][facei])
                            &(CfPatch[facei] - mesh_.C().boundaryField()[patchi][facei] - 0.5*xii*dt));
                        }
                        else 
                        {
                            gSurfPatch[facei] = 0.5*(
                                    iGbarPvol[faceCells[facei]] + 
                                    (   (iGbarPgrad[faceCells[facei]]) & (CfPatch[facei] - C[faceCells[facei]] - 0.5*xii*dt) )
                                    + gBarPvol_.BOUNDARY_FIELD_REF[patchi][facei] + 
                                    (   (gBarPgrad_.BOUNDARY_FIELD_REF[patchi][facei]) & (CfPatch[facei] - mesh_.C().boundaryField()[patchi][facei] - 0.5*xii*dt) )
                            );
                            hSurfPatch[facei] = 0.5*(
                                    iHbarPvol[faceCells[facei]] + 
                                    (   (iHbarPgrad[faceCells[facei]]) & (CfPatch[facei] - C[faceCells[facei]] - 0.5*xii*dt) )
                                    + hBarPvol_.BOUNDARY_FIELD_REF[patchi][facei] + 
                                    (   (hBarPgrad_.BOUNDARY_FIELD_REF[patchi][facei]) & (CfPatch[facei] - mesh_.C().boundaryField()[patchi][facei] - 0.5*xii*dt) )
                            );

                        }

                    }
                }
                else if (type == "symmetryPlane" || type == "DVMsymmetry")
                {
                    forAll(gSurfPatch, facei)
                    {
                        vector faceSf= SfPatch[facei];
                        if ((xii&faceSf) >  -VSMALL ) // outgoing and **reside(shouldn't be ignored)** DF, 
                                                    //incomming shoud be proceed after this function
                        {
                            gSurfPatch[facei] = iGbarPvol[faceCells[facei]] 
                            + ((iGbarPgrad[faceCells[facei]])
                            &(CfPatch[facei] - C[faceCells[facei]] - 0.5*xii*dt));
                            hSurfPatch[facei] = iHbarPvol[faceCells[facei]] 
                            + ((iHbarPgrad[faceCells[facei]])
                            &(CfPatch[facei] - C[faceCells[facei]] - 0.5*xii*dt));
                        } 
                    }
                }
            }
        }
    }
}


void Foam::fvDVM::updateMaxwellWallRho()
{
#if FOAM_MAJOR <= 3
    GeometricField<scalar, fvPatchField, volMesh>::GeometricBoundaryField& 
        rhoBCs = rhoVol_.boundaryField();
#else
    GeometricField<scalar, fvPatchField, volMesh>::Boundary& 
        rhoBCs = rhoVol_.BOUNDARY_FIELD_REF;
#endif
    forAll(rhoBCs, patchi)
    {
        if (rhoBCs[patchi].type() == "calculatedMaxwell")
        {
            calculatedMaxwellFvPatchField<scalar>& rhoPatch = 
                refCast<calculatedMaxwellFvPatchField<scalar> >(rhoBCs[patchi]);
            if(args_.optionFound("dvParallel"))
                mpiReducer_.reduceField(rhoPatch.outGoing());
        }
    }
    rhoVol_.correctBoundaryConditions();
}

void Foam::fvDVM::updateGHbarSurfMaxwellWallIn()
{
    #pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS)
    forAll(DV_, DVid){
        vector xii = XisGlobal[DVid];
        discreteVelocity& dv = DV_[DVid];
        surfaceScalarField& gSurf_ = *gSurf[DVid];
        surfaceScalarField& hSurf_ = *hSurf[DVid];
        forAll(gSurf_.BOUNDARY_FIELD_REF, patchi)
        {
            if (gSurf_.BOUNDARY_FIELD_REF[patchi].type() == "maxwellWall")
            {
                fvsPatchScalarField& gSurfPatch = gSurf_.BOUNDARY_FIELD_REF[patchi];
                fvsPatchScalarField& hSurfPatch = hSurf_.BOUNDARY_FIELD_REF[patchi];
                const fvPatchScalarField& rhoVolPatch = 
                    rhoVol_.BOUNDARY_FIELD_REF[patchi];
                const fvPatchVectorField& UvolPatch = 
                    Uvol_.BOUNDARY_FIELD_REF[patchi];
                const fvPatchScalarField& TvolPatch = 
                    Tvol_.BOUNDARY_FIELD_REF[patchi];
                const fvsPatchVectorField& SfPatch = 
                    mesh_.Sf().boundaryField()[patchi];
                forAll(gSurfPatch, facei)
                {
                    vector faceSf = SfPatch[facei];
                    if ((xii & faceSf) <= 0) // incomming
                    {
                        // set to maxwellian
                        gSurfPatch[facei] = rhoVolPatch[facei]
                        *dv.equilibriumMaxwellByRho
                            (
                                UvolPatch[facei],
                                TvolPatch[facei]
                            );

                        //set hSurf at maxwellWall to zero! , WRONG!!!
                        hSurfPatch[facei] = 
                            gSurfPatch[facei]*(R_.value()*TvolPatch[facei])
                        *(KInner_ + 3 - mesh_.nSolutionD());
                    }
                }
            }
        }
    }
}

void Foam::fvDVM::updateGHbarSurfSymmetryIn()
{
    //1. copy all DV's g/h to rho patch's dfContainer
    //2. MPI_Allgather the rho patch's dfContainer
    //if(args_.optionFound("dvParallel"))
    //{
    label rank  = mpiReducer_.rank();
    label nproc = mpiReducer_.nproc();
#if FOAM_MAJOR <= 3
    GeometricField<scalar, fvPatchField, volMesh>::GeometricBoundaryField& 
        rhoBCs = rhoVol_.boundaryField();
#else
    GeometricField<scalar, fvPatchField, volMesh>::Boundary& 
        rhoBCs = rhoVol_.BOUNDARY_FIELD_REF;
#endif
    forAll(rhoBCs, patchi)
    {
        label ps = rhoBCs[patchi].size();
        if (rhoBCs[patchi].type() == "symmetryMod")
        {
            symmetryModFvPatchField<scalar>& rhoPatch = 
                refCast<symmetryModFvPatchField<scalar> >(rhoBCs[patchi]);
            //compose the recvcout and displacement array
            labelField recvc(nproc);
            labelField displ(nproc);
            label chunck = nXi_/nproc;
            label left   = nXi_%nproc;
            forAll(recvc, i)
            {
                recvc[i] = 2*ps*(chunck + (i<left)) ;
                if(i<=left)
                    displ[i] = i*2*ps*(chunck + 1); // (i<=nXi_%nproc)
                else
                    displ[i] = 2*ps*(left*(chunck +1) + (i-left)*(chunck));
            }

            // check 12*28+15 dv's g
            label did = 1709;
            label pp  = did%nproc;
            label lid = did/nproc;
            //if(rank==pp)
            //{
                //Info << "processing by rank " << rank << endl;
                //Info << "12*28+15 outging g " << DV_[lid].gSurf()[0] << endl;
                //Info << "12*28+15 outging xi " << DV_[lid].xi() <<endl;
                //Info << "12*28+15 outging at boundary " << DV_[lid].gSurf().boundaryField()[patchi][0] << endl;
            //}
            // memcpy each dv's g/h to rho
            forAll(DV_, DVid)
            {
                //label shift = (nXi_ / nproc * rank + DVid)*2*ps;
                label shift = displ[rank] + DVid*2*ps;
                memcpy( (rhoPatch.dfContainer().data() + shift),
                        DV_[DVid].gSurf().boundaryField()[patchi].cdata(), ps*sizeof(scalar));
                memcpy( (rhoPatch.dfContainer().data() + shift + ps),
                        DV_[DVid].hSurf().boundaryField()[patchi].cdata(), ps*sizeof(scalar));
            }

            // check 
            //if(rank == pp)
                //Info << "dv gid 1709's g = " <<rhoPatch.dfContainer()[displ[pp]+lid*2*ps+32]<< endl;;


            //Allgather
            MPI_Allgatherv(
                //rhoPatch.dfContainer().data() + displ[rank],//2*ps*nXI_/nproc*rank, //send*
                MPI_IN_PLACE,
                2*ps*DV_.size(), //(how many DV i processed) * 2 * patch size
                MPI_DOUBLE,
                rhoPatch.dfContainer().data(),
                recvc.data(),
                displ.data(),
                MPI_DOUBLE,
                MPI_COMM_WORLD
                );
        }
    }
        forAll(DV_, DVid)
            DV_[DVid].updateGHbarSurfSymmetryIn();
}

void Foam::fvDVM::updateMacroSurf()
{
    // Init to zero before add one DV by one DV
    rhoSurf_ =  dimensionedScalar("0", rhoSurf_.dimensions(), 0);
    Usurf_ = dimensionedVector("0", Usurf_.dimensions(), vector(0, 0, 0));
    Tsurf_ = dimensionedScalar("0", Tsurf_.dimensions(), 0);
    qSurf_ = dimensionedVector("0", qSurf_.dimensions(), vector(0, 0, 0));
    stressSurf_ = dimensionedTensor
        (
            "0", 
            stressSurf_.dimensions(), 
            pTraits<tensor>::zero
        );

    surfaceVectorField rhoUsurf = rhoSurf_*Usurf_;
    surfaceScalarField rhoEsurf = rhoSurf_*magSqr(Usurf_);
    scalarField& parentrhoSurf_ = rhoSurf_;
    vectorField& parentrhoUsurf = rhoUsurf;
    scalarField& parentrhoEsurf = rhoEsurf;
    vectorField& parentqSurf_ = qSurf_;
    parentrhoSurf_ = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, dvSize), scalarField(faceSize,0),
    [&] (tbb::blocked_range<size_t> r, scalarField local_res) {
        for (size_t i = r.begin(); i < r.end(); i++) {
            local_res += weightsGlobal[i]*(static_cast<scalarField const>(*gSurf[i]));
        }
        return local_res;
    }, [] (scalarField x, scalarField y) {
        return x + y;
    });
    parentrhoUsurf = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, dvSize), vectorField(faceSize,{.0,.0,.0}),
    [&] (tbb::blocked_range<size_t> r, vectorField local_res) {
        for (size_t i = r.begin(); i < r.end(); i++) {
            local_res += weightsGlobal[i]*(static_cast<scalarField const>(*gSurf[i]))*XisGlobal[i];
        }
        return local_res;
    }, [] (vectorField x, vectorField y) {
        return x + y;
    });
    parentrhoEsurf = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, dvSize), scalarField(faceSize,0),
    [&] (tbb::blocked_range<size_t> r, scalarField local_res) {
        for (size_t i = r.begin(); i < r.end(); i++) {
            local_res += 0.5*weightsGlobal[i]*((static_cast<scalarField const>(*gSurf[i]))*magSqr(XisGlobal[i])+(static_cast<scalarField const>(*hSurf[i])));
        }
        return local_res;
    }, [] (scalarField x, scalarField y) {
        return x + y;
    });
    forAll(mesh_.boundary(),patchi)
    {
        forAll(DV_, dvi){
            rhoSurf_.BOUNDARY_FIELD_REF[patchi]  += weightsGlobal[dvi]*gSurf[dvi]->BOUNDARY_FIELD_REF[patchi];
            rhoUsurf.BOUNDARY_FIELD_REF[patchi]  += weightsGlobal[dvi]*gSurf[dvi]->BOUNDARY_FIELD_REF[patchi]*XisGlobal[dvi];
            rhoEsurf.BOUNDARY_FIELD_REF[patchi]  += 0.5*weightsGlobal[dvi]
            *(
                    gSurf[dvi]->BOUNDARY_FIELD_REF[patchi]*magSqr(XisGlobal[dvi]) 
                + hSurf[dvi]->BOUNDARY_FIELD_REF[patchi]
                );
        }
    }

    if(args_.optionFound("dvParallel"))
    {
        mpiReducer_.reduceField(rhoSurf_);
        mpiReducer_.reduceField(rhoUsurf);
        mpiReducer_.reduceField(rhoEsurf);
    }

    //- get Prim. from Consv.
    Usurf_ = rhoUsurf/rhoSurf_;

    Tsurf_ = (rhoEsurf - 0.5*rhoSurf_*magSqr(Usurf_))/((KInner_ + 3)/2.0*R_*rhoSurf_);

    updateTau(tauSurf_, Tsurf_, rhoSurf_);
    //- peculiar vel.

    // surfaceVectorField c = Usurf_;

    //-get part heat flux 
    // forAll(DV_, dvi)
    // {
    //     discreteVelocity& dv = DV_[dvi];
    //     c = dv.xi() - Usurf_;
    //     qSurf_ += 0.5*dXiCellSize_*dv.weight()*c
    //         *(
    //              magSqr(c)*dv.gSurf() 
    //            + dv.hSurf()
    //          );
    //     //- stressSurf is useless as we never update cell macro by macro flux 
    //     //- Comment out it as it is expansive
    //     //stressSurf_ += 
    //         //dXiCellSize_*dv.weight()*dv.gSurf()*c*c;
    // }
    parentqSurf_  = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, dvSize), vectorField(faceSize,{.0,.0,.0}),
    [&] (tbb::blocked_range<size_t> r, vectorField local_res) {
        for (size_t i = r.begin(); i < r.end(); i++) {
            vectorField c = XisGlobal[i] - static_cast<vectorField>(Usurf_);
            local_res += 0.5*weightsGlobal[i]*c
            *(
                 magSqr(c)*static_cast<scalarField const>(*gSurf[i])
               + static_cast<scalarField const>(*hSurf[i])
             );
        }
        return local_res;
    }, [] (vectorField x, vectorField y) {
        return x + y;
    });
    forAll(mesh_.boundary(),patchi)
    {
        forAll(DV_, dvi){
            vectorField c = XisGlobal[dvi] - Usurf_.BOUNDARY_FIELD_REF[patchi];
            qSurf_.BOUNDARY_FIELD_REF[patchi] += 0.5*weightsGlobal[dvi]*c
                *(
                    magSqr(c)*gSurf[dvi] ->BOUNDARY_FIELD_REF[patchi]
                + hSurf[dvi]->BOUNDARY_FIELD_REF[patchi]
                );
        }
    }

    //- correction for bar to original
    qSurf_ = 2.0*tauSurf_/(2.0*tauSurf_ + 0.5*time_.deltaT()*Pr_)*qSurf_;



    //- stress at surf is not used, as we dont't update macro in cell by macro flux at surface
    //stressSurf_ = 
        //2.0*tauSurf_/(2.0*tauSurf_ + 0.5*time_.deltaT())*stressSurf_;

    //- heat flux at wall is specially defined. as it ignores the velocity and temperature slip
    //- NOTE: To be changed as it is part macro, but it will not affect the innner fields, so we change it later
// #if FOAM_MAJOR <= 3
//     GeometricField<scalar, fvPatchField, volMesh>::GeometricBoundaryField& 
//         rhoBCs = rhoVol_.boundaryField();
// #else
//     GeometricField<scalar, fvPatchField, volMesh>::Boundary& 
//         rhoBCs = rhoVol_.BOUNDARY_FIELD_REF;
// #endif
//     qWall_ = dimensionedVector("0", qWall_.dimensions(), vector(0, 0, 0));
//     stressWall_ = dimensionedTensor
//         (
//             "0", 
//             stressWall_.dimensions(), 
//             pTraits<tensor>::zero
//         );
//     forAll(rhoBCs, patchi)
//     {
//         if (rhoBCs[patchi].type() == "calculatedMaxwell")
//         {
//             fvPatchField<vector>& qPatch = qWall_.BOUNDARY_FIELD_REF[patchi];
//             fvPatchField<vector>& Upatch = Uvol_.BOUNDARY_FIELD_REF[patchi];
//             fvPatchField<tensor>& stressPatch = stressWall_.BOUNDARY_FIELD_REF[patchi];
//             //- tau at surface use the tau at slip temperature as it is.
//             fvsPatchField<scalar>&  tauPatch = tauSurf_.BOUNDARY_FIELD_REF[patchi];
//             forAll(qPatch, facei)
//             {
//                 forAll(DV_, dvi)
//                 {
//                     scalar dXiCellSize = dXiCellSize_.value();
//                     discreteVelocity& dv = DV_[dvi];
//                     vector xi = dv.xi().value();
//                     vector c = xi - Upatch[facei];
//                     qPatch[facei] += 0.5*dXiCellSize*dv.weight()*c  //sometimes wall moves, then c != \xi
//                         *(
//                              magSqr(c)*dv.gSurf().boundaryField()[patchi][facei]
//                            + dv.hSurf().boundaryField()[patchi][facei]
//                          );
//                     stressPatch[facei] += 
//                         dXiCellSize*dv.weight()*dv.gSurf().boundaryField()[patchi][facei]*xi*xi;
//                 }
//                 qPatch[facei] = 2.0*tauPatch[facei]/(2.0*tauPatch[facei] + 0.5*time_.deltaT().value()*Pr_)*qPatch[facei];
//                 stressPatch[facei] = 
//                     2.0*tauPatch[facei]/(2.0*tauPatch[facei] + 0.5*time_.deltaT().value())*stressPatch[facei];
//             }
//             if(args_.optionFound("dvParallel"))
//             {
//                 mpiReducer_.reduceField(qPatch);
//                 mpiReducer_.reduceField(stressPatch);
//             }
//         }
//     }
}

void Foam::fvDVM::updateGHsurf()
{
    scalar dt = time_.deltaTValue();
    const scalarField &rho = rhoSurf_;
    const vectorField &U = Usurf_;
    const scalarField &T = Tsurf_;
    const vectorField &q = qSurf_;
    const scalarField &tau = tauSurf_;
    scalar R = R_.value();
    label D = mesh_.nSolutionD();
#pragma omp parallel num_threads(NUM_THREADS)
{
    #pragma omp for schedule(dynamic) nowait
    for(size_t dvi=0;dvi<dvSize;dvi++){
        for(size_t internal_id = 0;internal_id<faceSize;internal_id++){
            // if constexpr (vol){
            // relaxFactor = 1.5*dt/(2.0*tau[internal_id] + dt);
            // }else{
            scalar relaxFactor = 0.5*dt/(2.0*tau[internal_id] + 0.5*dt);
            // }

            scalar cSqrByRT = magSqr(U[internal_id]-XisGlobal[dvi])/(R*T[internal_id]);
            
            scalar cqBy5pRT = ((XisGlobal[dvi] - U[internal_id])& q[internal_id])/(5.0*rho[internal_id]*R*T[internal_id]*R*T[internal_id]);    
            scalar gEqBGK = rho[internal_id]/pow(sqrt(2.0*pi*R*T[internal_id]),D)*exp(-cSqrByRT/2.0);
                
            _gSurf[dvi*faceSize+internal_id] = (1.0 - relaxFactor)*_gSurf[dvi*faceSize+internal_id] + relaxFactor*( 1.0 + (1.0 - Pr_)*cqBy5pRT*(cSqrByRT - D - 2.0) )*gEqBGK;
            _hSurf[dvi*faceSize+internal_id] = (1.0 - relaxFactor)*_hSurf[dvi*faceSize+internal_id] + relaxFactor*( (KInner_ + 3.0 - D) + (1.0 - Pr_)*cqBy5pRT*((cSqrByRT - D)*(KInner_ + 3.0 - D) - 2*KInner_) )*gEqBGK*R*T[internal_id];
        }
    }
    #pragma omp for schedule(dynamic)
    for(size_t dvi=0;dvi<dvSize;dvi++){
        discreteVelocity& dv = DV_[dvi];
        vector xii = XisGlobal[dvi];
        surfaceScalarField& gSurf_ = dv.gSurf();
        surfaceScalarField& hSurf_ = dv.hSurf();
        volScalarField& gBarPvol_ = dv.gBarPvol();
        forAll(gSurf_.BOUNDARY_FIELD_REF, patchi)
        {
            scalarField cSqrByRT = static_cast<scalarField>(magSqr(Usurf_.boundaryField()[patchi] - xii)/(R*Tsurf_.boundaryField()[patchi]));

            scalarField cqBy5pRT  = static_cast<scalarField>(((xii - Usurf_.boundaryField()[patchi])&qSurf_.boundaryField()[patchi])/(5.0*rhoSurf_.boundaryField()[patchi]*R*Tsurf_.boundaryField()[patchi]*R*Tsurf_.boundaryField()[patchi]));

            scalarField gEqBGK  = static_cast<scalarField>(rhoSurf_.boundaryField()[patchi]/pow(sqrt(2.0*pi*R*Tsurf_.boundaryField()[patchi]),D)*exp(-cSqrByRT/2.0));

            fvsPatchScalarField& gSurfPatch = gSurf_.BOUNDARY_FIELD_REF[patchi];
            fvsPatchScalarField& hSurfPatch = hSurf_.BOUNDARY_FIELD_REF[patchi];
            // fvsPatchScalarField& gEqPatch   =    gEq.BOUNDARY_FIELD_REF[patchi];
            // fvsPatchScalarField& hEqPatch   =    hEq.BOUNDARY_FIELD_REF[patchi];

            scalarField gEqPatch   =    (( 1.0 + (1.0 - Pr_)*cqBy5pRT*(cSqrByRT - D - 2.0) )*gEqBGK);
            scalarField hEqPatch   =    (( (KInner_ + 3.0 - D) + (1.0 - Pr_)*cqBy5pRT*((cSqrByRT - D)*(KInner_ + 3.0 - D) - 2*KInner_) )*gEqBGK*R*T);

            const fvsPatchVectorField& SfPatch = mesh_.Sf().boundaryField()[patchi];

            scalarField relaxFactorPatch = 
                (0.5*dt/(2.0*tauSurf_.boundaryField()[patchi] + 0.5*dt));
            // fvsPatchScalarField& relaxFactorPatch = 
            //     relaxFactor.BOUNDARY_FIELD_REF[patchi];

            //NOTE:  If keep the below block, for the Sod problem, -parallel running would be different from serial run.
            //       I also doubt that for cyclic boundary, the same problem will happen.
            //       But if I totally deleted it, my experience shows for hypersonic cylinder flow, the free stream BC will have problem.
            //       So, I keep it, but exclude the cases of processor/cyclic/processorCyclic.
            //       The reason I guess is that, for those 'coupled' type boundary condition, the boundary field will be calculated when 
            //       doing GeometricField calculation, such as the relaxiation computation. While for those boundary condition I've defined, 
            //       such as mixed or maxwell, I have to explicitly calculate the unchanged boundary values.
            forAll(gSurfPatch, facei)
            {
                if ( (xii&(SfPatch[facei])) > 0    // Here, if delted , the free stream BC may bo s problem
                && gBarPvol_.BOUNDARY_FIELD_REF[patchi].type() != "processor"
                && gBarPvol_.BOUNDARY_FIELD_REF[patchi].type() != "processorCyclic"
                && gBarPvol_.BOUNDARY_FIELD_REF[patchi].type() != "cyclic")
                //&& gSurf_.BOUNDARY_FIELD_REF[patchi].type() != "DVMsymmetry")
                {
                    gSurfPatch[facei] = (1.0 - relaxFactorPatch[facei])
                        *gSurfPatch[facei]
                    + relaxFactorPatch[facei]*gEqPatch[facei];
                    hSurfPatch[facei] = (1.0 - relaxFactorPatch[facei])
                        *hSurfPatch[facei]
                    + relaxFactorPatch[facei]*hEqPatch[facei];
                }
            }
            //both in and out DF has been relaxed at DVMsymmetry boundary
            if(gSurfPatch.type() == "DVMsymmetry")
            {
                gSurfPatch = (1.0 - relaxFactorPatch)
                    *gSurfPatch
                + relaxFactorPatch*gEqPatch;
                hSurfPatch = (1.0 - relaxFactorPatch)
                    *hSurfPatch
                + relaxFactorPatch*hEqPatch;
            }
        }
    }
}


}

void Foam::fvDVM::updateGHtildeVol()
{
    #pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS)
    for(size_t dvi=0;dvi<dvSize;dvi++){
        discreteVelocity& dv = DV_[dvi];
        volScalarField& gTildeVol_ = dv.gTildeVol();
        volScalarField& hTildeVol_ = dv.hTildeVol();
        surfaceScalarField& gSurf_ = dv.gSurf();
        surfaceScalarField& hSurf_ = dv.hSurf();
        volScalarField& gBarPvol_ = dv.gBarPvol();
        volScalarField& hBarPvol_ = dv.hBarPvol();
        const labelUList& owner = mesh_.owner();
        const labelUList& neighbour = mesh_.neighbour();
        const vectorField Sf = mesh_.Sf();
        const scalarField V = mesh_.V();
        const scalar dt = time_.deltaTValue();
        const vector xii = XisGlobal[dvi];
        for(size_t internal_id = 0;internal_id<cellSize;internal_id++){
            gTildeVol_[internal_id] = -1.0/3*gTildeVol_[internal_id] + 4.0/3*gBarPvol_[internal_id];
            hTildeVol_[internal_id] = -1.0/3*hTildeVol_[internal_id] + 4.0/3*hBarPvol_[internal_id];
        }
        forAll(gSurf_.BOUNDARY_FIELD_REF, patchi){
            gTildeVol_.BOUNDARY_FIELD_REF[patchi] = -1.0/3*gTildeVol_.BOUNDARY_FIELD_REF[patchi]  + 4.0/3*gBarPvol_.BOUNDARY_FIELD_REF[patchi] ;
            hTildeVol_.BOUNDARY_FIELD_REF[patchi] = -1.0/3*hTildeVol_.BOUNDARY_FIELD_REF[patchi]  + 4.0/3*hBarPvol_.BOUNDARY_FIELD_REF[patchi] ;
        }

        // internal faces
        forAll(owner, facei)
        {
            const label own = owner[facei];
            const label nei = neighbour[facei];
            gTildeVol_[own] -= ((xii&Sf[facei])*gSurf_[facei]*dt/V[own]);
            gTildeVol_[nei] += ((xii&Sf[facei])*gSurf_[facei]*dt/V[nei]);
            hTildeVol_[own] -= ((xii&Sf[facei])*hSurf_[facei]*dt/V[own]);
            hTildeVol_[nei] += ((xii&Sf[facei])*hSurf_[facei]*dt/V[nei]);
        }

        // boundary faces
        forAll(gSurf_.BOUNDARY_FIELD_REF, patchi)
        {
            const fvsPatchField<scalar>& gSurfPatch =
                gSurf_.BOUNDARY_FIELD_REF[patchi];
            const fvsPatchField<scalar>& hSurfPatch =
                hSurf_.BOUNDARY_FIELD_REF[patchi];
            const fvsPatchField<vector>& SfPatch =
                mesh_.Sf().boundaryField()[patchi];
            const labelUList& pOwner = mesh_.boundary()[patchi].faceCells();
            forAll(pOwner, pFacei)
            {
                const label own = pOwner[pFacei];
                gTildeVol_[own] -= (xii&SfPatch[pFacei]) 
                *gSurfPatch[pFacei]*dt/V[own];
                hTildeVol_[own] -= (xii&SfPatch[pFacei]) 
                *hSurfPatch[pFacei]*dt/V[own];
            }
        }
    }
}

void Foam::fvDVM::updateMacroVol()
{
    //- Old macros, used only if we update using macro fluxes.
    volVectorField rhoUvol = rhoVol_*Uvol_;
    volScalarField rhoEvol = rhoVol_*(0.5*magSqr(Uvol_) + (KInner_ + 3)/2.0*R_*Tvol_);
    qVol_ = dimensionedVector("0", qVol_.dimensions(), vector(0, 0, 0));

    if(macroFlux_ == "no") // update cell macro by moment from DF
    {
        //- init to zeros
        rhoVol_ = dimensionedScalar("0", rhoVol_.dimensions(), 0);
        rhoUvol = dimensionedVector("0", rhoUvol.dimensions(), vector(0, 0, 0));
        rhoEvol = dimensionedScalar("0", rhoEvol.dimensions(), 0);

        //- get part macro
        forAll(DV_, dvi)
        {
            discreteVelocity& dv = DV_[dvi];
            rhoVol_ += dXiCellSize_*dv.weight()*dv.gTildeVol();
            rhoUvol += dXiCellSize_*dv.weight()*dv.gTildeVol()*dv.xi();
            rhoEvol += 0.5*dXiCellSize_*dv.weight()
                *(
                        magSqr(dv.xi())*dv.gTildeVol() 
                        + dv.hTildeVol()
                 );
        }
        //- get global macro via MPI_Allreduce
        if(args_.optionFound("dvParallel"))
        {
            mpiReducer_.reduceField(rhoVol_);
            mpiReducer_.reduceField(rhoUvol);
            mpiReducer_.reduceField(rhoEvol);
        }
    }
    else // update by macro flux
    {
        const labelUList& owner = mesh_.owner();
        const labelUList& neighbour = mesh_.neighbour();
        const vectorField Sf = mesh_.Sf();
        const scalarField V = mesh_.V();
        const scalar dt = time_.deltaTValue();
#ifdef VMESH
//init flux to zero
		rhoflux_ = dimensionedScalar("0", rhoVol_.dimensions(), 0);
		volVectorField rhouflux_= rhoflux_*Uvol_;
		volScalarField rhoeflux_ = rhoflux_ * magSqr(Uvol_);

		forAll(DV_, dvi)
		{    
			//temp var 
			rho_ = dimensionedScalar("0", rhoSurf_.dimensions(), 0);
			surfaceVectorField u_= rho_*Usurf_;
			surfaceScalarField e_ = rho_ * magSqr(Usurf_);

			discreteVelocity &dv = DV_[dvi];
			vector xii = dv.xi().value();

			rho_ = dXiCellSize_ * dv.weight() * dv.gSurf(); 
			u_ = dXiCellSize_ * dv.weight() * dv.gSurf() * dv.xi();
			e_  = 0.5*dXiCellSize_*dv.weight()
           *(
                dv.gSurf()*magSqr(dv.xi()) 
              + dv.hSurf()
            );
			
			//internal faces
			forAll(owner, facei)
			{
				const label own = owner[facei];
				const label nei = neighbour[facei];

				rhoflux_[own] -= ((xii & Sf[facei]) * rho_[facei] * dt / V[own]);
				rhoflux_[nei] += ((xii & Sf[facei]) * rho_[facei] * dt / V[nei]);
				rhouflux_[own] -= ((xii & Sf[facei]) * u_[facei] * dt / V[own]);
				rhouflux_[nei] += ((xii & Sf[facei]) * u_[facei] * dt / V[nei]);
				rhoeflux_[own] -= ((xii & Sf[facei]) * e_[facei] * dt / V[own]);
				rhoeflux_[nei] += ((xii & Sf[facei]) * e_[facei] * dt / V[nei]);
			}

			forAll(rhoSurf_.boundaryField(), patchi)
			{   
				const fvsPatchField<vector> &SfPatch = mesh_.Sf().boundaryField()[patchi];
				const labelUList &pOwner = mesh_.boundary()[patchi].faceCells();

				forAll(pOwner, pFacei)
				{
					const label own = pOwner[pFacei];
					rhoflux_[own] -= ((xii & SfPatch[pFacei]) * rho_.boundaryField()[patchi][pFacei] * dt / V[own]);
					rhouflux_[own] -= ((xii & SfPatch[pFacei]) * u_.boundaryField()[patchi][pFacei] * dt / V[own]);
					rhoeflux_[own] -= ((xii & SfPatch[pFacei]) * e_.boundaryField()[patchi][pFacei]  * dt / V[own]);
				}
			}
		}

        rhoVol_ += rhoflux_;
		rhoUvol += rhouflux_;
		rhoEvol += rhoeflux_;
    }
#else
        // internal faces
        #pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS)
        forAll(owner, facei)
        {
            const label own = owner[facei];
            const label nei = neighbour[facei];
            scalar rhoVol_sub = (rhoSurf_[facei]*Usurf_[facei]&Sf[facei])*dt/V[own];
            scalar rhoVol_add = (rhoSurf_[facei]*Usurf_[facei]&Sf[facei])*dt/V[nei];
            vector rhoUvol_sub = (rhoSurf_[facei]*Usurf_[facei]*Usurf_[facei] + stressSurf_[facei])&Sf[facei]*dt/V[own];
            vector rhoUvol_add = (rhoSurf_[facei]*Usurf_[facei]*Usurf_[facei] + stressSurf_[facei])&Sf[facei]*dt/V[nei];
            scalar rhoEsurf = rhoSurf_[facei] *(magSqr(Usurf_[facei]) + (KInner_ + 3)/2.0*R_.value()*Tsurf_[facei]);
            scalar rhoEvol_sub = (rhoEsurf*Usurf_[facei] + qSurf_[facei]) &Sf[facei]*dt/V[own];
            scalar rhoEvol_add = (rhoEsurf*Usurf_[facei] + qSurf_[facei]) &Sf[facei]*dt/V[nei];
            #pragma omp atomic
            rhoVol_[own] -= rhoVol_sub; 
            #pragma omp atomic
            rhoVol_[nei] += rhoVol_add; 
            #pragma omp atomic
            rhoEvol[own] -= rhoEvol_sub;
            #pragma omp atomic
            rhoEvol[nei] += rhoEvol_add;
            #pragma omp critical
            {
                rhoUvol[own] -= rhoUvol_sub;
                rhoUvol[nei] += rhoUvol_add; 
            }
        }
        // boundary faces
        forAll(rhoSurf_.boundaryField(), patchi)
        {
            const fvsPatchField<scalar>& rhoSurfPatch =
                rhoSurf_.boundaryField()[patchi];
            const fvsPatchField<vector>& UsurfPatch =
                Usurf_.boundaryField()[patchi];
            const fvsPatchField<scalar>& TsurfPatch =
                Tsurf_.boundaryField()[patchi];
            const fvsPatchField<tensor>& stressSurfPatch =
                stressSurf_.boundaryField()[patchi];
            const fvsPatchField<vector>& qSurfPatch =
                qSurf_.boundaryField()[patchi];
            const fvsPatchField<vector>& SfPatch =
                mesh_.Sf().boundaryField()[patchi];

            const labelUList& pOwner = mesh_.boundary()[patchi].faceCells();
            forAll(pOwner, pFacei)
            {
                const label own = pOwner[pFacei];
                rhoVol_[own] -= (rhoSurfPatch[pFacei]*UsurfPatch[pFacei]
                        &SfPatch[pFacei])*dt/V[own];
                rhoUvol[own] -= (rhoSurfPatch[pFacei]*UsurfPatch[pFacei]*UsurfPatch[pFacei]
                        + stressSurfPatch[pFacei])&Sf[pFacei]*dt/V[own];
                scalar rhoEsurf = 
                    rhoSurfPatch[pFacei]
                    *(magSqr(UsurfPatch[pFacei]) + (KInner_ + 3)/2.0*R_.value()*TsurfPatch[pFacei]);

                rhoEvol[own] -= (rhoEsurf*UsurfPatch[pFacei] + qSurfPatch[pFacei])
                    &SfPatch[pFacei]*dt/V[own];
            }
        }
    }
#endif

    //- get Prim. from Consv.
    Uvol_ = rhoUvol/rhoVol_;
    Tvol_ = (rhoEvol - 0.5*rhoVol_*magSqr(Uvol_))/((KInner_ + 3)/2.0*R_*rhoVol_);

    //- Correct the macro field boundary conition
    Uvol_.correctBoundaryConditions();
    Tvol_.correctBoundaryConditions();
    //- Note for maxwell wall, the operation here update 
    //- the boundary rho field but it's meaningless.

    //- The vol macro field's boundary field is meanless!
    //rhoVol_.correctBoundaryConditions(); 

    //- update tau
    updateTau(tauVol_, Tvol_, rhoVol_);
    //- peculiar vel.
    // volVectorField c = Uvol_;

    //-get part heat flux
    vectorField& parentqVol_ = qVol_;
    parentqVol_  = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, dvSize), vectorField(cellSize,{.0,.0,.0}),
    [&] (tbb::blocked_range<size_t> r, vectorField local_res) {
        for (size_t i = r.begin(); i < r.end(); i++) {
            vectorField c = XisGlobal[i] - static_cast<vectorField>(Uvol_);
            local_res += 0.5*weightsGlobal[i]*c
            *(
                 magSqr(c)*static_cast<scalarField const>(*gTildeVol[i])
               + static_cast<scalarField const>(*hTildeVol[i])
             );
        }
        return local_res;
    }, [] (vectorField x, vectorField y) {
        return x + y;
    });
    forAll(mesh_.boundary(),patchi)
    {
        forAll(DV_, dvi){
            vectorField c = XisGlobal[dvi] - Uvol_.BOUNDARY_FIELD_REF[patchi];
            qVol_.BOUNDARY_FIELD_REF[patchi] += 0.5*weightsGlobal[dvi]*c
                *(
                    magSqr(c)*gTildeVol[dvi] ->BOUNDARY_FIELD_REF[patchi]
                + hTildeVol[dvi]->BOUNDARY_FIELD_REF[patchi]
                );
        }
    }
    // forAll(DV_, dvi)
    // {
    //     discreteVelocity& dv = DV_[dvi];
    //     c = dv.xi() - Uvol_;
    //     qVol_ += 0.5*dXiCellSize_*dv.weight()*c
    //         *(
    //                 magSqr(c)*dv.gTildeVol() 
    //                 + dv.hTildeVol()
    //          );
    // }

    //- get global heat flux via MPI_Allreduce
    if(args_.optionFound("dvParallel"))
        mpiReducer_.reduceField(qVol_);
    //- correction for bar to original
    qVol_ = 2.0*tauVol_/(2.0*tauVol_ + time_.deltaT()*Pr_)*qVol_;
}

void Foam::fvDVM::updatePressureInOutBC()
{
    // for pressureIn and pressureOut BC, the boundary value of Uvol(in/out) and Tvol(in/out) should be updated here!
    // boundary faces
#if FOAM_MAJOR <= 3
    GeometricField<scalar, fvPatchField, volMesh>::GeometricBoundaryField& 
        rhoBCs = rhoVol_.boundaryField();
#else
    GeometricField<scalar, fvPatchField, volMesh>::Boundary& 
        rhoBCs = rhoVol_.BOUNDARY_FIELD_REF;
#endif
    forAll(rhoBCs, patchi)
    {
        if (rhoBCs[patchi].type() == "pressureIn")
        {
            const fvsPatchField<vector>& SfPatch = mesh_.Sf().boundaryField()[patchi];
            const fvsPatchField<scalar>& magSfPatch = mesh_.magSf().boundaryField()[patchi];
            pressureInFvPatchField<scalar>& rhoPatch = 
                refCast<pressureInFvPatchField<scalar> >(rhoBCs[patchi]);
            fvPatchField<vector>& Upatch = Uvol_.BOUNDARY_FIELD_REF[patchi];
            const fvPatchField<scalar>& Tpatch = Tvol_.boundaryField()[patchi];
            const scalar pressureIn = rhoPatch.pressureIn();
            // now changed rho and U patch
            const labelUList& pOwner = mesh_.boundary()[patchi].faceCells();
            forAll(rhoPatch, facei)
            {
                const scalar  Tin = Tpatch[facei];
                // change density
                rhoPatch[facei] = pressureIn/R_.value()/Tin; // Accturally not changed at all :p

                // inner boundary cell data state data
                label own = pOwner[facei];
                vector Ui = Uvol_[own];
                scalar Ti = Tvol_[own];
                scalar rhoi = rhoVol_[own];
                scalar ai = sqrt(R_.value() * Ti * (KInner_ + 5)/(KInner_ + 3)); // sos

                // change normal velocity component based on the characteristics
                vector norm = SfPatch[facei]/magSfPatch[facei]; // boundary face normal vector
                scalar Un = Ui & norm; // normal component
                scalar UnIn = Un + (pressureIn - rhoi * R_.value() * Ti)/rhoi/ai; // change normal component
                Upatch[facei] = UnIn * norm + (Ui - Un * norm); // tangential component not changed.
            }
        }
        else if(rhoBCs[patchi].type() == "pressureOut")
        {
            const fvsPatchField<vector>& SfPatch = mesh_.Sf().boundaryField()[patchi];
            const fvsPatchField<scalar>& magSfPatch = mesh_.magSf().boundaryField()[patchi];
            pressureOutFvPatchField<scalar>& rhoPatch = 
                refCast<pressureOutFvPatchField<scalar> >(rhoBCs[patchi]);
            fvPatchField<vector>& Upatch = Uvol_.BOUNDARY_FIELD_REF[patchi];
            fvPatchField<scalar>& Tpatch = Tvol_.BOUNDARY_FIELD_REF[patchi];
            const scalar pressureOut = rhoPatch.pressureOut();
            // now changed rho and U patch
            const labelUList& pOwner = mesh_.boundary()[patchi].faceCells();
            forAll(rhoPatch, facei)
            {
                // inner cell data state data
                label own = pOwner[facei];
                vector Ui = Uvol_[own];
                scalar Ti = Tvol_[own];
                scalar rhoi = rhoVol_[own];
                scalar ai = sqrt(R_.value() * Ti * (KInner_ + 5)/(KInner_ + 3)); // sos

                // change outlet density
                rhoPatch[facei] = rhoi  +  (pressureOut - rhoi * R_.value() * Ti)/ai/ai; // Accturally not changed at all :p
                Tpatch[facei] = pressureOut/(R_.value() * rhoi);

                // change normal velocity component based on the characteristics
                vector norm = SfPatch[facei]/magSfPatch[facei]; // boundary face normal vector
                scalar Un = Ui & norm; // normal component
                scalar UnIn = Un + ( rhoi * R_.value() * Ti - pressureOut)/rhoi/ai; // change normal component
                Upatch[facei] = UnIn * norm + (Ui - Un * norm); // tangential component not changed.
            }
        }
    }
}

template<template<class> class PatchType, class GeoMesh>
void Foam::fvDVM::updateTau
(
 GeometricField<scalar, PatchType, GeoMesh>& tau,
 const GeometricField<scalar, PatchType, GeoMesh>& T, 
 const GeometricField<scalar, PatchType, GeoMesh>& rho
 )
{
    tau = muRef_*exp(omega_*log(T/Tref_))/rho/T/R_;
}


void Foam::fvDVM::writeDFonCell(label cellId)
{
    std::ostringstream convert;
    convert << cellId;
    scalarIOList df
    (
        IOobject
        (
             "DF"+convert.str(),
             "0",
             mesh_,
             IOobject::NO_READ,
             IOobject::AUTO_WRITE
        )
    );
    //set size of df
    df.setSize(nXi_);

    scalarList dfPart(DV_.size());
    //put cellId's DF to dfPart
    forAll(dfPart, dfi)
        dfPart[dfi] = DV_[dfi].gTildeVol()[cellId];

    label nproc = mpiReducer_.nproc();
    //gather
    //tmp list for recv
    scalarList dfRcv(nXi_);

    //Compose displc and recvc
    labelField recvc(nproc);
    labelField displ(nproc);
    label chunck = nXi_/nproc;
    label left   = nXi_%nproc;
    forAll(recvc, i)
    {
        recvc[i] = chunck + (i<left) ;
        if(i<=left)
            displ[i] = i*(chunck + 1); // (i<=nXi_%nproc)
        else
            displ[i] = left*(chunck +1) + (i-left)*(chunck);
    }
    MPI_Gatherv(dfPart.data(), dfPart.size(), MPI_DOUBLE,
            dfRcv.data(), recvc.data(), displ.data(),
            MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //reposition
    if(mpiReducer_.rank() == 0)
    {
        forAll(df, i)
        {
            label p   = i%nproc;
            label ldi = i/nproc;
            df[i] = dfRcv[displ[p]+ldi];
        }
        df.write();
    }
}

void Foam::fvDVM::writeDFonCells()
{
    if (time_.outputTime())
        forAll(DFwriteCellList_, i)
            writeDFonCell(i);
}

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

    Foam::fvDVM::fvDVM
(
 volScalarField& rho,
 volVectorField& U,
 volScalarField& T,
 int* argc,
 char*** argv,
 Foam::argList& args
 )
    :
        IOdictionary
        (
         IOobject
         (
          "DVMProperties",
          T.time().constant(),
          T.mesh(),
          IOobject::MUST_READ,
          IOobject::NO_WRITE
         )
        ),
        mesh_(rho.mesh()),
        time_(rho.time()),
        rhoVol_(rho),
        Uvol_(U),
        Tvol_(T),
        args_(args),
        fvDVMparas_(subOrEmptyDict("fvDVMparas")),
        gasProperties_(subOrEmptyDict("gasProperties")),
        nXiPerDim_(readLabel(fvDVMparas_.lookup("nDV"))),
        xiMax_(fvDVMparas_.lookup("xiMax")),
        xiMin_(fvDVMparas_.lookup("xiMin")),
        dXi_((xiMax_-xiMin_)/(nXiPerDim_ - 1)),
        dXiCellSize_
        (
         "dXiCellSize",
         pow(dimLength/dimTime, 3),
         scalar(1.0)
        ),
        macroFlux_(fvDVMparas_.lookupOrDefault("macroFlux", word("no"))),
        //res_(fvDVMparas_.lookupOrDefault("res", 1.0e-12)),
        //checkSteps_(fvDVMparas_.lookupOrDefault("checkSteps", 100)),
        R_(gasProperties_.lookup("R")),
        omega_(readScalar(gasProperties_.lookup("omega"))),
        Tref_(gasProperties_.lookup("Tref")),
        muRef_(gasProperties_.lookup("muRef")),
        Pr_(readScalar(gasProperties_.lookup("Pr"))),
        KInner_((gasProperties_.lookupOrDefault("KInner", 0))),
        mpiReducer_(args, argc, argv), // args comes from setRootCase.H in dugksFoam.C;
        DV_(0),
        rhoSurf_
        (
         IOobject
         (
          "rhoSurf",
          mesh_.time().timeName(),
          mesh_,
          IOobject::NO_READ,
          IOobject::NO_WRITE
         ),
         mesh_,
         dimensionedScalar( "0", rho.dimensions(), 0)
        ),
        rhoflux_
        (
            IOobject
            (
                "rhoflux",
                mesh_.time().timeName(),
                mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            mesh_,
            dimensionedScalar("0", rho.dimensions(), 0)
        ),
        rho_
        (
            IOobject
            (
                "rhotemp",
                mesh_.time().timeName(),
                mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            mesh_,
            dimensionedScalar("0", rho.dimensions(), 0)
        ),
        Tsurf_
        (
         IOobject
         (
          "Tsurf",
          mesh_.time().timeName(),
          mesh_,
          IOobject::NO_READ,
          IOobject::NO_WRITE
         ),
         mesh_,
         dimensionedScalar( "0", T.dimensions(), 0)
        ),
        Usurf_
        (
         IOobject
         (
          "Usurf",
          mesh_.time().timeName(),
          mesh_,
          IOobject::NO_READ,
          IOobject::NO_WRITE
         ),
         mesh_,
         dimensionedVector( "0", U.dimensions(), vector(0,0,0))
        ),
        qSurf_
        (
         IOobject
         (
          "qSurf",
          mesh_.time().timeName(),
          mesh_,
          IOobject::NO_READ,
          IOobject::AUTO_WRITE
         ),
         mesh_,
         dimensionedVector( "0", dimMass/pow(dimTime,3), vector(0,0,0))
        ),
        stressSurf_
        (
         IOobject
         (
          "stressSurf",
          mesh_.time().timeName(),
          mesh_,
          IOobject::NO_READ,
          IOobject::AUTO_WRITE
         ),
         mesh_,
         dimensionedTensor("0", dimensionSet(1,-1,-2,0,0,0,0), pTraits<tensor>::zero)
        ),
        qVol_
        (
         IOobject
         (
          "q",
          mesh_.time().timeName(),
          mesh_,
          IOobject::NO_READ,
          IOobject::AUTO_WRITE
         ),
         mesh_,
         dimensionedVector( "0", dimMass/pow(dimTime,3), vector(0,0,0))
        ),
        tauVol_
        (
         IOobject
         (
          "tauVol",
          mesh_.time().timeName(),
          mesh_,
          IOobject::NO_READ,
          IOobject::NO_WRITE
         ),
         mesh_,
         dimensionedScalar( "0", dimTime, 0)
        ),
        tauSurf_
        (
         IOobject
         (
          "tauSurf",
          mesh_.time().timeName(),
          mesh_,
          IOobject::NO_READ,
          IOobject::NO_WRITE
         ),
         mesh_,
         dimensionedScalar( "0", dimTime, 0)
        ),
        qWall_
        (
         IOobject
         (
          "qWall",
          mesh_.time().timeName(),
          mesh_,
          IOobject::NO_READ,
          IOobject::AUTO_WRITE
         ),
         mesh_,
         dimensionedVector( "0", dimMass/pow(dimTime,3), vector(0,0,0))
        ),
        stressWall_
        (
         IOobject
         (
          "stressWall",
          mesh_.time().timeName(),
          mesh_,
          IOobject::NO_READ,
          IOobject::AUTO_WRITE
         ),
         mesh_,
         dimensionedTensor("0", dimensionSet(1,-1,-2,0,0,0,0), pTraits<tensor>::zero)
        ),
        init(NUM_THREADS)
{
    DFwriteCellList_ = lookupOrDefault<labelList>("DFwriteCellList", labelList()),
    initialiseDV();
    setCalculatedMaxwellRhoBC();
    setSymmetryModRhoBC();
    // set initial rho in pressureIn/Out BC
    updatePressureInOutBC();
    updateTau(tauVol_, Tvol_, rhoVol_); //calculate the tau at cell when init
    Usurf_ = fvc::interpolate(Uvol_, "linear"); // for first time Dt calculation.
}

// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::fvDVM::~fvDVM()
{
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::fvDVM::evolution()
{
    // Info << "Begin evolution" << endl;
    TICK(updateGHbarPvol)
    updateGHbarPvol();
    TOCK(updateGHbarPvol)
    TICK(GHbarPvolComm)
    GHbarPvolComm();
    TOCK(GHbarPvolComm)
    TICK(updateGrad)
    updateGrad();
    TOCK(updateGrad)
    // Info << "Done updateGHbarPvol " << endl;
    TICK(updateGHbarSurf)
    updateGHbarSurf();
    TOCK(updateGHbarSurf)
    // Info << "Done updateGHbarSurf " << endl;
    TICK(updateMaxwellWallRho)
    updateMaxwellWallRho();
    TOCK(updateMaxwellWallRho)
    // Info << "Done updateMaxwellWallRho " << endl;
    TICK(updateGHbarSurfMaxwellWallIn)
    updateGHbarSurfMaxwellWallIn();
    TOCK(updateGHbarSurfMaxwellWallIn)
    // Info << "Done updateGHbarSurfMaxwellWallIn " << endl;
    TICK(updateGHbarSurfSymmetryIn)
    updateGHbarSurfSymmetryIn();
    TOCK(updateGHbarSurfSymmetryIn)
    // Info << "Done updateGHbarSurfSymmetryIn " << endl;
    TICK(updateMacroSurf)
    updateMacroSurf();
    TOCK(updateMacroSurf)
    // Info << "Done updateMacroSurf " << endl;
    TICK(updateGHsurf)
    updateGHsurf();
    TOCK(updateGHsurf)
    // Info << "Done updateGHsurf " << endl;
    TICK(updateGHtildeVol)
    updateGHtildeVol();
    TOCK(updateGHtildeVol)
    // Info << "Done updateGHtildeVol " << endl;
    TICK(updateMacroVol)
    updateMacroVol();
    TOCK(updateMacroVol)
    //
    updatePressureInOutBC();
}


void Foam::fvDVM::getCoNum(scalar& maxCoNum, scalar& meanCoNum)
{
    scalar dt = time_.deltaTValue();
    scalarField UbyDx =
        mesh_.surfaceInterpolation::deltaCoeffs()
        *(mag(Usurf_) + sqrt(scalar(mesh_.nSolutionD()))*xiMax_);
    maxCoNum = gMax(UbyDx)*dt;
    meanCoNum = gSum(UbyDx)/UbyDx.size()*dt;
}


const fieldMPIreducer& Foam::fvDVM::mpiReducer() const
{
    return mpiReducer_;
}


// ************************************************************************* //
