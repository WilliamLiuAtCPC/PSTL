# PSTL
 
This is a PyTorch Implementation of our proposed PSTL method.





Prerequisites:

1. Download Cholec80 dataset at:http://camma.u-strasbg.fr/datasets
2. Download TestSamples at：https://pan.baidu.com/s/1pSurJrfEva37-a02dwrWvA password：973g
3. Packages in "requirements.txt" should be installed before running our code.




Usage:

1. You can directly use our model to perform surgery tool detection tasks by running "testModel.py".
You can find our pretrained model using the link given in "/pretrained model checkpoint/download link.txt".
In order to perform PSTL on your dataset, you have to build your dataset according to our TestSamples. Don't forget to modify the filepath used in our code according to your local enviorment.

2. You can run "trainRegressor.py" to train a specific regressor on your bounding box training set.
To generate pseudo bounding box annotation, you may need the implementation of DDT listed below:
https://github.com/GeoffreyChen777/DDT

3. You can run "/ACW/acw_training.py" to train a specific classifier on your tool classification training set.
"loadmodel_test.py" is designed for classifier testing. You can change the parameter used in this file to test your classifier.
Here is a Tool for manual crop task: https://github.com/WilliamLiuAtCPC/PyCut-WL

4. If you'd like to design a new Dataset structure for your specific dataset, you can modify "/data/mydataset.py" according to your requirement.
/*"noboxfuse.py""fasterrcnn_only.py" are just the implementation of our ablation study. Ignore them if you don't need.*/




Reference:
[1] our paper