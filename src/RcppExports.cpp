// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// gammaLogLikelihood
double gammaLogLikelihood(double x, double shape, double rate);
RcppExport SEXP _tagmReDraft_gammaLogLikelihood(SEXP xSEXP, SEXP shapeSEXP, SEXP rateSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type x(xSEXP);
    Rcpp::traits::input_parameter< double >::type shape(shapeSEXP);
    Rcpp::traits::input_parameter< double >::type rate(rateSEXP);
    rcpp_result_gen = Rcpp::wrap(gammaLogLikelihood(x, shape, rate));
    return rcpp_result_gen;
END_RCPP
}
// invGammaLogLikelihood
double invGammaLogLikelihood(double x, double shape, double scale);
RcppExport SEXP _tagmReDraft_invGammaLogLikelihood(SEXP xSEXP, SEXP shapeSEXP, SEXP scaleSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type x(xSEXP);
    Rcpp::traits::input_parameter< double >::type shape(shapeSEXP);
    Rcpp::traits::input_parameter< double >::type scale(scaleSEXP);
    rcpp_result_gen = Rcpp::wrap(invGammaLogLikelihood(x, shape, scale));
    return rcpp_result_gen;
END_RCPP
}
// wishartLogLikelihood
double wishartLogLikelihood(arma::mat X, arma::mat V, double n, arma::uword P);
RcppExport SEXP _tagmReDraft_wishartLogLikelihood(SEXP XSEXP, SEXP VSEXP, SEXP nSEXP, SEXP PSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type V(VSEXP);
    Rcpp::traits::input_parameter< double >::type n(nSEXP);
    Rcpp::traits::input_parameter< arma::uword >::type P(PSEXP);
    rcpp_result_gen = Rcpp::wrap(wishartLogLikelihood(X, V, n, P));
    return rcpp_result_gen;
END_RCPP
}
// invWishartLogLikelihood
double invWishartLogLikelihood(arma::mat X, arma::mat Psi, double nu, arma::uword P);
RcppExport SEXP _tagmReDraft_invWishartLogLikelihood(SEXP XSEXP, SEXP PsiSEXP, SEXP nuSEXP, SEXP PSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Psi(PsiSEXP);
    Rcpp::traits::input_parameter< double >::type nu(nuSEXP);
    Rcpp::traits::input_parameter< arma::uword >::type P(PSEXP);
    rcpp_result_gen = Rcpp::wrap(invWishartLogLikelihood(X, Psi, nu, P));
    return rcpp_result_gen;
END_RCPP
}
// mvtLogLikelihood
double mvtLogLikelihood(arma::vec x, arma::vec mu, arma::mat Sigma, double nu);
RcppExport SEXP _tagmReDraft_mvtLogLikelihood(SEXP xSEXP, SEXP muSEXP, SEXP SigmaSEXP, SEXP nuSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type mu(muSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Sigma(SigmaSEXP);
    Rcpp::traits::input_parameter< double >::type nu(nuSEXP);
    rcpp_result_gen = Rcpp::wrap(mvtLogLikelihood(x, mu, Sigma, nu));
    return rcpp_result_gen;
END_RCPP
}
// runSemiSupervisedMDI
Rcpp::List runSemiSupervisedMDI(arma::uword R, arma::uword thin, arma::field<arma::mat> Y, arma::uvec K, arma::uvec types, arma::umat labels, arma::umat fixed);
RcppExport SEXP _tagmReDraft_runSemiSupervisedMDI(SEXP RSEXP, SEXP thinSEXP, SEXP YSEXP, SEXP KSEXP, SEXP typesSEXP, SEXP labelsSEXP, SEXP fixedSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::uword >::type R(RSEXP);
    Rcpp::traits::input_parameter< arma::uword >::type thin(thinSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::mat> >::type Y(YSEXP);
    Rcpp::traits::input_parameter< arma::uvec >::type K(KSEXP);
    Rcpp::traits::input_parameter< arma::uvec >::type types(typesSEXP);
    Rcpp::traits::input_parameter< arma::umat >::type labels(labelsSEXP);
    Rcpp::traits::input_parameter< arma::umat >::type fixed(fixedSEXP);
    rcpp_result_gen = Rcpp::wrap(runSemiSupervisedMDI(R, thin, Y, K, types, labels, fixed));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_tagmReDraft_gammaLogLikelihood", (DL_FUNC) &_tagmReDraft_gammaLogLikelihood, 3},
    {"_tagmReDraft_invGammaLogLikelihood", (DL_FUNC) &_tagmReDraft_invGammaLogLikelihood, 3},
    {"_tagmReDraft_wishartLogLikelihood", (DL_FUNC) &_tagmReDraft_wishartLogLikelihood, 4},
    {"_tagmReDraft_invWishartLogLikelihood", (DL_FUNC) &_tagmReDraft_invWishartLogLikelihood, 4},
    {"_tagmReDraft_mvtLogLikelihood", (DL_FUNC) &_tagmReDraft_mvtLogLikelihood, 4},
    {"_tagmReDraft_runSemiSupervisedMDI", (DL_FUNC) &_tagmReDraft_runSemiSupervisedMDI, 7},
    {NULL, NULL, 0}
};

RcppExport void R_init_tagmReDraft(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
