#pragma once
class vectorFieldReducer{
    public:

    std::vector<surfaceScalarField*>& gSurf;
    scalarField& weightsGlobal;
    vectorField& XisGlobal;
    vectorField rhoUsurf;
    int faceSize;
    vectorFieldReducer(std::vector<surfaceScalarField*>& gSurf,scalarField& weightsGlobal,vectorField& XisGlobal,int faceSize)
        :gSurf(gSurf),weightsGlobal(weightsGlobal),XisGlobal(XisGlobal),faceSize(faceSize),rhoUsurf(faceSize,{0,0,0}){}

    vectorFieldReducer( vectorFieldReducer& reducer, tbb::split )
        :gSurf(reducer.gSurf),weightsGlobal(reducer.weightsGlobal),XisGlobal(reducer.XisGlobal),faceSize(reducer.faceSize),rhoUsurf(reducer.faceSize,{0,0,0}){}

    void operator()(const tbb::blocked_range<size_t>& range) {
        for (size_t i = range.begin(); i != range.end(); ++i) {
            rhoUsurf += weightsGlobal[i]*(static_cast<scalarField const>(*gSurf[i]))*XisGlobal[i];
        }
    }
    void join(const vectorFieldReducer& other) {
            rhoUsurf += other.rhoUsurf;
    }
};
