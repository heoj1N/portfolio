@echo off
echo Running all match_plots_task.py files...
echo.

echo Running K-Nearest Neighbors...
python "01_K_Nearest_Neighbors/match_plots_task.py"
pause

echo Running Decision Trees...
python "02_Decision_Trees/match_plots_task.py"
pause

echo Running Probabilistic Inference...
python "03_Probabilistic_Inference/match_plots_task.py"
pause

echo Running Linear Regression...
python "04_Linear_Regression/match_plots_task.py"
pause

echo Running Linear Classification...
python "05_Linear_Classification/match_plots_task.py"
pause

echo Running Optimization...
python "06_Optimization/match_plots_task.py"
pause

echo Running Deep Learning...
python "07-08_Deep_Learning/match_plots_task.py"
pause

echo Running SVM and Kernels...
python "09_SVM_and_Kernels/match_plots_task.py"
pause

echo Running Dimensionality Reduction...
python "10_Dimensionality_Reduction_and_Matrix_Factorization/match_plots_task.py"
pause

echo Running Clustering...
python "11_Clustering/match_plots_task.py"
pause

echo All match_plots_task.py files have been run. 