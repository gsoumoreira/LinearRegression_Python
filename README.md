## Linear Regression
Linear regression using python

## Linear regression using python

Regression is a statistical analysis designed to understand the behavior of a sample data. This analysis was first proposed by Francis Galton in his publication: Regression Towards Mediocrity in Hereditary Stature, in the nineteenth century, in which he proposed that the size of pea seeds, whose ancestors had large stature, tended to regress to an average.
Linear regression is the regression of a series of data to a linear function. Or in other words, a linear function that represents a series of data. In this repository, we will use data from Coursera's course Machine Learning (Professor Andrew Ng from Stanford University), as well as follow the development steps of code and theory proposed by the course itself. However, python will be the programming language used, with the purpose of exercising the development of linear regression in another language, as well as discussing other possible paths that python libraries enable

## About the exercises

These are two exercises proposed. The first (ex1.py): Suppose you own a famous food truck franchise. You will be presented with a table showing the population size of each city where the food trucks are located (column 2) and their respective profits (column 1). Apply linear regression and check if there is any correlation between these informations. If so, estimate the profit of the food truck for a population of 35,000 people and then for 70,000 people.
The second (ex2.py): Suppose you are a resident of the of Portland city and you want to sell your house, but you do not know for sure the price to be chosen. You are presented with a table with the size, number of rooms and the price of a collection of houses in the same city. Apply linear regression and check if there is any correlation between these informations.

```python
pathtodata = 'Exercise_Data/ex1_Data.txt'
data = pd.read_csv(pathtodata,delimiter = ',',header=None)
```
