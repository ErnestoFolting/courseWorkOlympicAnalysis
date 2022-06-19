from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns
import pandas as p
import numpy as n

def read_dataset(path, separ, en):
    return p.read_csv(path, sep = separ, encoding = en)

#Info and head output
def info_head(dataset, n):
    dataset.info()
    print(dataset.head(n))

#Drap column list in dataset
def drop_columns(dataset, columns):
    for column in columns:
        dataset = dataset.drop(column, axis = 1)
    return dataset

#Adding column to dataset
def add_column(data, column, condition):
    data[column] = condition

#Tests for normality
def kolm_smirn_check(dFrame, column):
    ks_statistic, p_value = st.kstest(dFrame[column], "norm")
    if p_value > 0.05: return True
    return False

def pearson_check(dFrame, column):
    statistic, p_value = st.normaltest(dFrame[column])
    if p_value > 0.05: return True
    return False

#Testing all columns for normality
def columns_normal_test(dataset):
    print("\n#1 test - Kolmagorov-Smirnov test,\n#2 test - Pearson test\n")
    for c in dataset.columns:
        print("{0:<15}".format(c + ":"),f"#1 test - {kolm_smirn_check(dataset, c)}, #2 test - {pearson_check(dataset, c)};")

#Correlation heatmap
def corr_plot(datasetCorr):
    sns.heatmap(datasetCorr.corr(),annot=True,lw=1,cmap = "Purples")

#splittin data to train and test data
def split_data(x, y):
    return train_test_split(x, y, test_size=0.3)

#Creating linear and polynomial models
def lin_pol_reg_models(x_train, y_train, par):
    lin = []
    pol = []
    for i in range(len(par)):
        pol.append(make_pipeline(PolynomialFeatures(degree = 2), LinearRegression()))
        if i < 4: 
            lin.append(LinearRegression().fit(x_train[par[i]].to_numpy().reshape(-1,1), y_train))
            pol[i].fit(x_train[par[i]].to_numpy().reshape(-1,1), y_train)
        else:
            lin.append(LinearRegression().fit(x_train[par[i]].values, y_train.values))
            pol[i].fit(x_train[par[i]].values, y_train.values)
    return lin, pol

#Creating predictions by models
def predict_by(x_test, par, prediction, s):
    for i in range(len(par)):
        if i < 4:
            prediction.append(s[i].predict(x_test[par[i]].to_numpy().reshape(-1,1)))
        else:
            prediction.append(s[i].predict(x_test[par[i]].values))
    return prediction

def make_predictions(x_test, par, lin, pol):
    prediction = []
    prediction = predict_by(x_test, par, prediction, lin)
    prediction = predict_by(x_test, par, prediction, pol)
    return prediction

#Flattening list 
def flatten_par(par):
    params = []
    for el in par:
        if (isinstance(el, list)):
            params.append(", ".join(el))
        else:
            params.append(el)
    return params

#Searching best model by mse
def find_best_by_mse(prediction, y_test, par):
    params = flatten_par(par)
    MSE = []
    for i in range(len(prediction)):
        tempPred = prediction[i]
        MSE.append(mean_squared_error(y_test,tempPred))
    min_i = n.argmin(MSE)
    print("\nBest model is:")
    if (min_i < 15):
        print(f"linear model with {params[min_i]} param(s)")
    else:
        print(f"polynomial model with {params[min_i - 15]} param(s)")
    return MSE, min_i

#Creating mse comparasion plot
def mse_plot(MSE, par):
    params = flatten_par(par)
    plt.figure(figsize=(10,7))  
    plt.plot(params, MSE[0:15], color = "green")
    plt.plot(params, MSE[15:30], color = "red")
    plt.legend(["Linear", "Polinomial"])
    plt.title("Comparison of MSE of linear and polynomial regressions")
    plt.xlabel("Parameters")
    plt.ylabel("MSE")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

#Creating polynomial prediction 2d plot
def pol_2d_plot(x_test, y_test, pred, column):
    mymodel = n.poly1d(n.polyfit(x_test[column], pred , 2))
    myline = n.linspace(0, 600, 600)
    plt.scatter(x_test[column], y_test)
    plt.plot(myline, mymodel(myline), color = "red")
    plt.title(f"Polynomial prediction Medal ~ {column}")
    plt.ylabel("Medals")
    plt.xlabel(column)
    plt.show()

#Creating polynomial prediction 3d plot
def pol_3d_plot(x_test, y_test, pol, params):
    params_values = []
    for i in range(len(params)):
        values = n.linspace(x_test[params[i]].min(), x_test[params[i]].max()).reshape(-1, 1)
        params_values.append(values)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')
    X_3d = params_values[0]
    Y_3d = params_values[1]
    XX, YY = n.meshgrid(X_3d, Y_3d)
    Z = []
    for i in range(len(Y_3d)):
        temp = []
        for j in range(len(X_3d)):
            temp.append(pol.predict(n.array([X_3d[j], Y_3d[i]]).T)[0])
        Z.append(temp)
    Z = n.array(Z)
    ax.set_title(f'Polynomial prediction Medals ~ {params[0]}, {params[1]}')
    ax.set_xlabel(params[0])
    ax.set_ylabel(params[1])
    ax.set_zlabel('Medals')
    ax.plot_surface(
        XX, YY,
        n.array(Z),
        color = "green",
        alpha=0.5
    )
    ax.scatter(x_test[params[0]], x_test[params[1]], y_test)
    plt.show()

#Comparing general avarage and prediction model avarage
def check_avarage(dataset, predictions, par):
    params = ', '.join(par) if (isinstance(par, list)) else par
    sum = 0
    count = 0
    for i in range(len(dataset)):
        count += 1
        sum += dataset.iloc[i, 4]
    print(f"\navarage medals amount:\n{round(sum / count, 5)}")
    sum = 0
    count = 0
    for pred in predictions:
        count += 1
        sum += pred
    print(f"avarage medals amount by polynomial model with {params} param(s):\n{round(sum / count, 5)}")

def main():
    dataset = read_dataset("fullDataset.csv", ",", "cp1252")
    info_head(dataset, 5)
    dataset = drop_columns(dataset, ["Unnamed: 0", "Sex", "Age", "Height", "Weight", "Team", "Sport"])
    add_column(dataset, "GDP per capita", dataset["GDP"] / dataset["Population"])
    info_head(dataset, 10)
    #grouping dataset by years, seasons and country codes
    dataset = dataset.groupby(["Year", "Season", "Code"]).agg(GDP = ("GDP", "mean"),
                                                              Population = ("Population", "mean"),
                                                              Members = ("ID","nunique"),
                                                              GDP_p_c = ("GDP per capita", "mean"),
                                                              Medal = ("Medal", "count"))
    info_head(dataset, 10)
    columns_normal_test(dataset)
    corr_plot(dataset.drop("Medal", axis = 1))
    x_train, x_test, y_train, y_test = split_data(dataset.iloc[:, :4], dataset["Medal"])
    par = ["GDP", "Population", "Members", "GDP_p_c",
          ["GDP", "Population"],["GDP", "Members"],["GDP", "GDP_p_c"], ["Members", "Population"], ["Members", "GDP_p_c"], ["Population", "GDP_p_c"],
          ["GDP", "Population","Members"], ["GDP", "Population", "GDP_p_c"], ["GDP", "Members", "GDP_p_c"], ["Population", "Members", "GDP_p_c"],
          ["GDP", "Population", "Members", "GDP_p_c"]]
    lin, pol = lin_pol_reg_models(x_train, y_train, par)
    prediction = make_predictions(x_test, par, lin, pol)
    MSE, min_i = find_best_by_mse(prediction, y_test, par)
    mse_plot(MSE, par)
    pol_2d_plot(x_test, y_test, prediction[17], par[2])
    pol_3d_plot(x_test, y_test, pol[7], par[7])
    pol_3d_plot(x_test, y_test, pol[8], par[8])
    if min_i < 15: check_avarage(dataset, prediction[min_i], par[min_i])
    else: check_avarage(dataset, prediction[min_i], par[min_i - 15])

if __name__ == "__main__":
    main()