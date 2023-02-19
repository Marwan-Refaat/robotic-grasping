from utils.dataset_processing.grasp import GraspRectangles #Import grasp.py file from utils -> dataset prcoessing

#Init GraspRectangles class
grasps = GraspRectangles()


"""
Loads grasps from cornell file which contains a 4x4x2 array of points, representing 4 grasp rectangles, 
returns 4 GraspRectangle objects
"""
grasps = grasps.load_from_cornell_file("cornell/08/pcd0800cpos.txt") 

print(vars(grasps[0]))

#Plots all grasps on matplotlib plot, takes shape of image ([height,width]) as arg
grasps.show(shape=[480,640])  


