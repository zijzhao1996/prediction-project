# Regression Project

This is code repo for a regression prediction project. 

## Usage

```python inference.py '20200123.yx.csv' 'Ridge_y1.pkl' 'Ridge_y2.pkl'```

Arguments: `.csv`: Any unseen data in current folder which structure is similar to the training data. (consist of X, Y1 and Y2)
           
           `.pkl`: Models that you want to load. You need load pickles file for both Y1 and Y2.

We have trained and uploaded all the models in the `\models` folder. You are free to choose any of those models.
1. Ridge_y1, Ridge_y2, 
2. KRR_y1, KRR_y2, 
3. RF_y1, RF_y2, 
3. GBoost_y1, GBoost_y2, 
4. LGBM_y1, LGBM_y2, 
5. Averaged_y1, Averaged_y2, 
6. Stacking_y1, Stacking_y2

## Outputs

A sample output is shown as below:

```-------- Loading the dataset sucessfully. -------
-------- Preprocessing the dataset sucessfully. -------
-------- Loading the models sucessfully. -------
The correlation coefficent between Y1 and Y1_pred is: 0.41719016444521495
The correlation coefficent between Y2 and Y2_pred is: 0.6901208720898381
The mean squared error between Y1 and Y1_pred is: 5.839785006652536e-06
The mean squared error between Y2 and Y2_pred is: 6.168002241185126e-06
Plot of predicted and ground truth Y values has been generated.
-------- Finished --------
```

You can easily check the correlation coefficent and mean squared error.
