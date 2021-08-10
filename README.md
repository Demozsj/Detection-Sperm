Data set name：MIaMIA-SpermVideo data set (MIaMIA-SVDS)

Data set release date: 2021.07.29

Last Updated: 2021.07.29

Database administrator: 
	Chen Li, Email: lichen201096@hotmail.com；
	Shuojia Zou, Email: 1599586845@qq.com；
	Ao Chen, Email: 982438710@qq.com；

Stakeholders: Chen Li, Shuojia Zou, Peng Xu, Ao Chen, Jiawei Zhang, Haoyuan Chen, Weiming Hu, Wanli, Liu,
	     Peng Zhao, Hechen Yang, Jindong Li, Xialin Li, Wenwei Zhao, Yilin Cheng.

Introduction to data Sets:
	This data set consists of three parts:  Subset-A provides more than 125,000 objects with bounding box annotation with category
	information of 101 videos for tiny object detection tasks; Subset-B segments more than 26,000 sperms in 10 videos as
	ground truth (GT) for tiny object tracking tasks; Subset-C provides more than 125,000 independent images of sperms and
	impurities for tiny object classification tasks.

Download the way:
	This is just a simple example of a data set due to github's limited data volume, which can be downloaded from https://doi.org/10.6084/m9.figshare.15074253.v1.

Use requirement:
	Any non-commercial research work is welcome to use the database.

Training model：main.py

Obtain test results: get_predicted_result.py

Get comment information: get_GroundTruth.py

The relevant evaluation indexes were calculated: calculate_mAP.py
