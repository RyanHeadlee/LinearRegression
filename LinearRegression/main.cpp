/* Ideas for consideration to implement:
[*] Regularization: Implement regularization techniques such as L1 (Lasso) or L2 (Ridge) regularization to prevent overfitting and improve generalization.

[*] Polynomial Regression: Extend the class to support polynomial regression by adding methods to generate polynomial features of different degrees.

[] Feature Scaling: Implement methods for feature scaling (e.g., normalization or standardization) to improve the convergence speed of optimization algorithms.

[] Cross-Validation: Add methods to perform k-fold cross-validation for model evaluation and hyperparameter tuning.

[*] Model Evaluation Metrics: Include methods to calculate various evaluation metrics besides mean squared error, such as R-squared, mean absolute error, or root mean squared error.

[] Batch Gradient Descent: Extend gradient descent optimization to support batch gradient descent, mini-batch gradient descent, or stochastic gradient descent for training large datasets efficiently.

[] Feature Selection: Implement feature selection techniques to identify the most relevant features for the regression task, such as forward selection, backward elimination, or recursive feature elimination.

[] Visualization: Add visualization methods to plot learning curves, convergence trajectories, or model predictions against actual values.

[*] Parallelization: Explore parallelization techniques (e.g., using multi-threading or parallel processing libraries) to speed up computationally intensive tasks, such as gradient descent optimization or cross-validation.

[] Serialization: Implement methods to serialize and deserialize the trained model to/from disk, allowing users to save and load models for later use without retraining.
*/

#include "LinearRegression.h"
#include <iostream>

int main() {

    std::vector<std::vector<double>> X = {
        {0.2745260821241825, 0.8565343230025708, 0.7491166674910877, 0.6081038881587227, 0.5842545113721088, 0.8386558049518352, 0.2873348372433737, 0.1279047505881754, 0.5096375276797437, 0.7561056080086631},
        {0.6620315895818508, 0.768990182788586, 0.7623805987269301, 0.5820875668654498, 0.045159576304555915, 0.167896563225993, 0.9208394557224254, 0.7801553138263734, 0.5773200545232259, 0.2996879891981681},
        {0.8537461468109023, 0.8385418195125294, 0.9029557187458764, 0.9521392150613466, 0.7345828523343255, 0.8482176404981938, 0.5366476448595364, 0.4637408792053149, 0.1289229541844228, 0.5990280662390879},
        {0.3796130509110057, 0.7394679083516449, 0.7300776047617837, 0.6519530178956158, 0.6588737161656631, 0.3997802344468635, 0.06287458239857224, 0.24174491415106225, 0.2717876905580925, 0.5168024828116806},
        {0.929785525457403, 0.6237242097700821, 0.4930267002490657, 0.4375666328346637, 0.4203361300252072, 0.30281525210114224, 0.3135089633764888, 0.8731571205384112, 0.23694153823176822, 0.9178676864814364},
        {0.42018084423562774, 0.9316301376603858, 0.837309057325383, 0.5467790450788973, 0.6947294746800601, 0.625458688038932, 0.182247358422987, 0.635099905074921, 0.4790579318808116, 0.3098332062198882},
        {0.14608284351470868, 0.70327207346384, 0.824012364922604, 0.829623777974112, 0.398355075708117, 0.06106571127150706, 0.0012438521524328339, 0.4372347650724233, 0.2664855223518461, 0.8218778263482127},
        {0.6010006496216941, 0.19813490014197, 0.056892733582410315, 0.6403168211364498, 0.5808773832702825, 0.2624955590080691, 0.6995113699922612, 0.7637361258235033, 0.27055734702107304, 0.028709822994309908},
        {0.4303933142838485, 0.5263352582065634, 0.569635841933371, 0.7678548219994927, 0.2996009784893616, 0.08445031652990109, 0.4832892375220591, 0.9008024881079443, 0.23836110964311452, 0.2991424657348753},
        {0.26236814299230496, 0.7877275363931956, 0.9394453270572145, 0.7784313991772174, 0.04201515778670645, 0.9055110350523175, 0.20968121894993206, 0.03236934971645627, 0.1643047480373133, 0.5263494720651333},
        {0.6891097161968833, 0.10125780180280865, 0.9723099676604493, 0.23455418751209785, 0.3420874922329759, 0.05480869834093991, 0.15791440269470606, 0.2773715290877632, 0.43833429901414246, 0.4407393707436189},
        {0.7537850111373724, 0.0371396840348987, 0.22979886675095682, 0.6516413126878292, 0.15000803822516244, 0.7271232485850057, 0.5941527369375981, 0.2747369561228954, 0.08981081065250135, 0.5331075393999481},
        {0.6757353610933478, 0.42120532704005895, 0.4954963024146362, 0.3638974370586184, 0.28742279218775313, 0.5914465322593494, 0.2660762370283356, 0.3001696935009561, 0.5208329395145982, 0.9765450087466427},
        {0.6565721578778762, 0.8871537344889924, 0.25476512165540244, 0.6563579176085312, 0.19189258272330334, 0.4684417212054982, 0.2348546785555219, 0.8890314808130069, 0.4255042485982331, 0.431028837543571},
        {0.8616355596385753, 0.7183369869491831, 0.8424350741404284, 0.7034585274078983, 0.7947316343673874, 0.11925464251630957, 0.09564451626284232, 0.3779742707321756, 0.9236159872953004, 0.2645988166305997},
    };
    std::vector<double> y = {
        0.3207053770559216, 0.34330602588944614, 0.12583586865947938, 0.8754743442635501, 0.7952784676291824,
        0.6174161820003987, 0.6066214261326082, 0.4027289221720869, 0.16757704067292, 0.485184461891622,
        0.8797124964631607, 0.04933659863060247, 0.5256318964780478, 0.14818494338017252, 0.8559932713772018,
    };

    std::vector<std::vector<double>> X_test = {
        {0.47812792438267393, 0.05085383883328698, 0.19553549162350636, 0.9060423805979921, 0.24381555688326928, 0.8148029733222667, 0.4231381345068361, 0.5948825403803611, 0.31757313501227047, 0.8683540876297416},
        {0.32473092650936086, 0.32639922366895346, 0.06289129088056111, 0.4314015907597589, 0.46941370304786156, 0.7196705307036004, 0.7226842236354224, 0.19336984329483997, 0.6904411868594892, 0.2529527617181191},
        {0.04510135549371053, 0.48444072408007643, 0.06788694363046233, 0.8582483445931429, 0.631850012650734, 0.2044358844659554, 0.7268452530572271, 0.4605848374981093, 0.8679805128396716, 0.37907035259500443},
        {0.2959821435252772, 0.5006127607397716, 0.6635200471554585, 0.2564626094138067, 0.002978035218279773, 0.39412310952418395, 0.9826636564157126, 0.6509380192453137, 0.3343884696739017, 0.9193050597979182},
        {0.902740986091061, 0.04669619998026463, 0.32808011918812456, 0.9229356806172644, 0.15234216501268572, 0.4419624077847717, 0.12747032466384803, 0.10173954870793338, 0.874116585373957, 0.08066712073089157}
    };
    std::vector<double> y_test = { 0.6247794677496661, 0.24302759313963973, 0.6657514467440689, 0.26542962690152594, 0.3906311248672244 };

    LinearRegression model;
    double alpha = 5e-5;
    int iterations = 10000;
    double lambda = 5;
    int degree = 1;
    model.fit(X, y, alpha, iterations, lambda, degree);

    std::vector<double> predictions;
    predictions = model.predict(X_test);

    std::cout << model.meanSquaredError(y_test, predictions) << '\n';
    std::cout << model.rSquared(y_test, predictions) << '\n';
    std::cout << model.rootMSError(y_test, predictions) << '\n';

    return 0;
}