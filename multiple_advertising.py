'''
多元回归——广告收入数据分析
1、数据及分析对象
'Advertising.csv'，数据集包含了200个不同市场的产品销售额，
每个销售额对应3种广告媒体投入成本，分别是：TV, radio, 和 newspaper

2、目的及分析任务
理解机器学习方法在数据分析中的应用——多元回归方法进行回归分析：
1) 进行数据预处理，绘制'TV','radio','newspaper'这三个自变量与因变量'sales'之间的相关关系图；
2) 采用两种不同方法进行多元回归分析——统计学方法和机器学习方法；
3) 进行模型预测，得出模型预测结果；
4) 对预测结果进行评价。

3.方法及工具
pandas, seaborn, matplotlib, statsmodels, scikit-learn 
'''
#%%              1.业务理解
'''
本例题涉及不同媒体的广告投入与产品销量额之间的关系。
该业务的主要内容是建立TV, radio, newspaper与sales之间的多元回归模型。
'''
#%%              2.数据读取
import pandas as pd
data = pd.read_csv('D:/desktop/ML/回归分析/Advertising.csv')
print(data)
#将'Number'这列删除
data = data.drop('Number',axis = 1)
data
#%%              3.数据理解
#%%% 探索性分析
#先查看变量之间的相关性
import seaborn as sns
import matplotlib.pyplot as plt
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')

#用pairplot()，绘制TV、radio、newspaper与sales之间的关系图
sns.pairplot(data, x_vars=['TV', 'radio', 'newspaper'], y_vars='sales', height=5, aspect=0.8,kind = 'reg')
plt.suptitle('Pairplot of Features vs Sales')

#%%              4.建模
X = data[['TV', 'radio', 'newspaper']]
Y = data['sales']
import statsmodels.api as sm
X = sm.add_constant(X)

model = sm.OLS(Y, X)
result = model.fit()
print(result.summary())
# =============================================================================
#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                  sales   R-squared:                       0.897
# Model:                            OLS   Adj. R-squared:                  0.896
# Method:                 Least Squares   F-statistic:                     570.3
# Date:                Mon, 28 Oct 2024   Prob (F-statistic):           1.58e-96
# Time:                        22:22:52   Log-Likelihood:                -386.18
# No. Observations:                 200   AIC:                             780.4
# Df Residuals:                     196   BIC:                             793.6
# Df Model:                           3                                         
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const          2.9389      0.312      9.422      0.000       2.324       3.554
# TV             0.0458      0.001     32.809      0.000       0.043       0.049
# radio          0.1885      0.009     21.893      0.000       0.172       0.206
# newspaper     -0.0010      0.006     -0.177      0.860      -0.013       0.011
# ==============================================================================
# Omnibus:                       60.414   Durbin-Watson:                   2.084
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):              151.241
# Skew:                          -1.327   Prob(JB):                     1.44e-33
# Kurtosis:                       6.332   Cond. No.                         454.
# ==============================================================================

# =============================================================================
# summary()方法的内容较多，其中重点考虑参数 R-squared、Prob(F-statistic)以及P>|t|的两个值，
# 通过这4个参数就能判断模型是否是线性显著的，同时知道显著的程度。
# R方=0.897，接近1，说明回归效果好。
# F检验的值越大越能推翻原假设（模型不是线性模型），Prob(F-statistic)值越小越能拒绝原假设，1.58e-96接近于0，说明模型是线性显著。
# =============================================================================

#%%                5.机器学习
#%%%先将数据集分割成训练集和验证集，用sklearn.model_selection中的train_test_split()来拆分，训练集75%，验证集25%，随机种子设为1
X = data[['TV', 'radio', 'newspaper']]
Y = data['sales']
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y , test_size=0.25, random_state=1)

#%%% 创建和拟合线性回归模型
from sklearn.linear_model import LinearRegression
linear_sk = LinearRegression()
model_sk = linear_sk.fit(X_train, Y_train) #训练模型

#%%% 查看多元线性回归模型的回归系数
model_sk.coef_      
#array([0.04656457, 0.17915812, 0.00345046])

#%%% 查看回归模型的截距
model_sk.intercept_
#2.87696662

#%%               6.模型预测
Y_pred = linear_sk.predict(X_test)
from sklearn.metrics import r2_score
print("R-squared:", r2_score(Y_test, Y_pred))
# R-squared: 0.9156213613792232
#%%% 或者调用score()，返回预测的R方，即模型准确率
model_sk.score(X_test, Y_test)
#0.91562136

#%%               7.模型评价
#画折线图来比较预测值和真实值的差距
import matplotlib.pyplot as plt
plt.plot(Y_pred, label = 'predict')
plt.plot(range(len(Y_pred)), Y_test, label = 'test')
plt.legend(loc='upper right')
plt.xlabel('the number of sales')
plt.ylabel('values of sales')

##预测结果与真实值的折线趋于重合，说明模型的预测结果较好。

#%%% 或者
plt.scatter(Y_test, Y_pred)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")

##如果模型预测准确，点应该接近对角线（y=x），即预测值应与实际值接近。

















