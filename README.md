ColorBot using Reinforcement Learning

Introduction:
Problem Statement:
Image colorization is conversion of a grayscale image to fully colored image by comparing with the
ground truth. Colorization of a black white photograph in present age is done by using Photoshop and a picture can take up to a month to colorize, taking into consideration the period it was taken in, the culture, region, flow of light and many other factors. Coloring a face alone requires up to 20 layers of pink, green and blue shades to get it right. It requires extensive research and a lot of manpower and can usually not be done without the supervision of a human. There should human interaction in order to specify the colors of the objects in the image. The requirement of human interaction problem was slowly overcome by using neural networks to solve this problem, but even then the amount of data required to train the model will be extensive.

Motivation:
Colorization techniques are used in MRI, black and white image restoration and it is a cumbersome task. There are existing methods using which the colorization is done by neural network, SVMs but all these models require extensive training demanding huge dataset. The motivation of this was to try and formulate a reinforcement learning approach which does not require human interaction and would reward itself by comparing to the ground truth of the detected object.

Review of other researches :
The following three papers in the domain of image colorization were reviewed - First one is “Deep Colorization by Zezhou Cheng, Qingxiong Yang, Bin Sheng”, this paper attempts to solve the colorization problem by using Neural Network. Second one is “Digital Image Colorization using Machine Learning” by Cris Zanoci and Jim Andress”, this paper attempts to solve the colorization problem by using SVMs and the feature they use to colorize the image is the texture of images.
Third one was “Colorful Image Colorization by Richard Zhang, Phillip Isola, Alexei A. Efros”, this paper attempts to solve the colorization problem by using a CNN. 

Open questions in the domain:
The colorization problem is quite well explored in the neural networks domain where large database  of image is available. But a way in which a machine can be trained without using a large database is yet to be explored!

short summary of your proposed approach:
In this project, we attempted to automate this process and create a ColorBot using reinforcement learning. In reinforcement learning, since it is reward based, it can give itself rewards by just working on a single image and training itself to color similar images. In this project, the agent works with general shapes and the colors for these general shape which is its ground truth. The agent learns to detect the shape and color it according to its original shade.

Backgrounds:
While image colorization is a boutique computer graphics task, it is also an instance of a difficult pixel prediction problem in computer vision. Here we have shown that colorization with a deep CNN and a well-chosen objective function can come closer to producing results indistinguishable from real color photos. Our method not only provides a useful graphics output, but can also be viewed as a pretext task for representation learning. Although only trained to color, our network learns a representation that is surprisingly useful for object classification, detection, and segmentation, performing strongly compared to other self-supervised pre-training methods.


“Deep Colorization by Zezhou Cheng,Qingxiong Yang, Bin Sheng”
In this paper the authors propose to perform colorization using Deep Neural network. The overview of the method followed by them is as follows: 
-	Train the DNN with different clusters of images. Here the authors initially group the images into various clusters using their proposed adaptive images clustering technique.
-	For a given test grayscale image the nearest cluster and corresponding trained DNN will be explored automatically first. The feature descriptors will be extracted at each pixel and serve as the input of the neural network.
-	The output is the chrominance of the corresponding pixel which can be directly combined with the luminance (grayscale pixel value) to obtain the corresponding color value. 
Limitations of this method:
-	It is supposed to be trained on a huge reference image database which contains all possible objects. This is impossible in practice. 
-	It is also impossible to recover the color information lost due to color to grayscale transformation. 

“Digital Image Colorization using Machine Learning” by Cris Zanoci and Jim Andress”
 
In this paper the author proposes to colorize the image by training a collection of SVMs.
Proposed method:
-	Input a single colored image to train on and a grayscale image of similar content.
-	The algorithm begins by creating a high dimensional representations found in the colored image.
-	After training a collection of SVMs on the input image, the color transfer process is phrased out as a energy minimization problem.
-	Approach tries to reconcile two competing goals of predicting the best color given the local texture and maintaining spatial consistency in predictions of the model.
Limitations:
-	Algorithm does poorly on images with smooth textures, like those of human faces and cloth, in which it often attributes one color to the entire image.

“Colorful Image Colorization by Richard Zhang, Phillip Isola, Alexei A. Efros”
In this paper the author proposes to colorize the image by training a CNN.
Proposed method:
-	They have proposed to use CNN to map from a grayscale input to a distribution over quantized color value outputs using a certain architecture designed by them.
-	Their focus is on the design of the objective function, and technique for inferring point estimates of color from the predicted color distribution.
Limitations:
-	This method also needs a large database of images to train the neural network.

Methods:
To achieve the goal of this project, there were few processes that had to be done prior to the colorization operation.
1.	Generate the dataset: This project works with a dataset that contains basic shapes, namely, triangle, rectangle, square and circle. So, to generate a dataset, the initial attempt was to have just one object in the image and for a procedure using neural network to detect it. This was the mid-term progress, where the dataset was available on kaggle but later on as the project progressed, the requirement of having multiple images in the same image was a difficult task. To overcome this, a program was created to generate these shapes with random colors but the generation of shapes was inefficient. Due to the problem of creating a dataset, the neural network procedure was replaced by detecting the shape using the python cv2 module. This program proved efficient to not require huge training dataset to learn to detect the shape.
2.	Type of objects in the image: The image now had multiple objects and was now exposed to two cases, overlapping and non-overlapping images. The non overlapping images could easily be dealt with using the procedure to detect shape but in the case of overlapping objects, the detection of images became inconvenient. This demanded for a procedure of erosion in order to separate the overlapping images.
3.	Decision of colorspace: When considering the color space of real world, the range would be extremely large to accommodate. In order to facilitate this problem, the definition of the color of each of objects were limited on to the three pure colors and no intermediate shades.
4.	Defining the reward function: In the mid-progress report the training of the image using neural network, helped decide the color that must be assigned to the object easier but later the agent never had the opportunity to train as much and thus wasn’t getting the required reward to term that particular state as the goal state. So, the reward function was modified accordingly to help allow the agent to learn the shade of the color space better.
5.	Maintenance of Q-table: Since the project works with different shapes and thus requires different q-tables for each of them, the individual q-tables were stored in the form of files which could be written onto during training and read from during testing. 
Algorithms:
Image processing:
This algorithm aims at processing the image fed and extracting the contours required from the image.
Input: Grayscale Image
Output: Contours of the objects in the image
1.	Read the image, resize it to width 300 and calculate the ratio of the image and the resized image.
2.	If the passed image is not a grayscale image, then it is converted to grayscale, slightly blurred and the thresholded. If the image overlaps then it is eroded. Now, the contours from this image is found

Shape detection:
This algorithm takes in the contours and returns the shape.
Input: Contours
Output: Shape of the contour
1.	The perimeter of the detected contours is found and the polygon is approximated which returns the vertices.
2.	The vertices that occur close by are neglected because these are the ones that are caused to due to inequalities in the texture of the objects.
3.	Now the length of the approximation is calculated and checked. If it is 3, it is a triangle, 4 is a rectangle else it’ll be a circle.

Reinforcement learning agent:
The agent is put into an environment that has an image with multiple objects and each object has a predefined color that it has to figure out and color.
Input: Grayscale image, contour
Output: Colored image
1.	The agent first receives the shape from the shape detector and performs training for that shape alone. 
2.	The agent tries to assign different values for the (R,G,B) tuple and gets a predefined reward where it gets 10 point if the shade is right, -10 if the shade is off and 30 if it is in the goal state.
3.	Once the agent is trained, the agent is then tested by feeding another image.


Experiments:
The project revolves around two types of images, overlapping and non-overlapping images. Experiments were conducted for three cases: non-overlapping, overlapping and in the scenario of noise. By noise, it means that the objects weren’t properly created or their edges weren’t defined.
1.	Non-overlapping images:

The accuracy of the model was tested on non-overlapping images which implies that the objects in the image do not have any overlapping or touching objects. The model was also tested if it only works for grayscale images or for even colored images. First, according to the description, a grayscale image was fed to the model:

                 
Figure 1: grayscale input(Non-overlapping)	Figure2: colored output(Non-overlapping)
The image in figure 1 was fed to the model and  shapes were detected in the first part. Once the shapes were detected it was then colored using the agent and the figure 2 shows the output.
 
Figure 3: detection of shape
2.	Overlapping images:
In the overlapping case, the image must first be processed with erosion so that the model can be in the position to detect the shapes. So a grayscale image with overlapping objects was given as shown in figure 4, where the triangle overlaps the circle.

          
     Figure 4:grayscale(overlapping)		      Figure 5: Eroded		      Figure 6: output(overlapping)
Once the image has been eroded to the following form(figure 5), it can then follow the same procedure and color it according to the rules specified. Figure 6 displays the output. The model was also tested on images where the overlapping of objects in the images were higher than 40% to check if the image processing technique works. This is shown in figure 7 and figure 8:
                                             
	Figure 7: input for overlapping over 40%			Figure 8: Output for overlapping over 40%
3.	Images with Noise:
After the testing of overlapping and non-overlapping images, experiments were also done to find out how the model performed when encountered with noise. Figure 9 has shapes along with noise in the image and this is how the model performed:
                                                           
		Figure 8: Input with noise				Figure 9: Output with noise
It was observed that the objects with noise was assumed to circle and colored according to the detected shapes shown in figure 10.
 
Figure 10: detection of shapes with noise
Conclusions:
Colorization of basic shapes using reinforcement learning was accomplished using image processing and reinforcement learning. The existing algorithms using neural networks are much advanced in the terms of the input images they process but this project uses a different idea of reinforcement learning for colorization and it can further be expanded by having better object detection functions and reward functions. 
This project was very educational and we learnt reinforcement learning in depth and also working with images has helped us to understand the complexities that may arise. 

Response to the feedback:
1. “Don't forget detailed procedure in final report”:
The image preprocessing procedure is explained under the methods sections.

2. “Just nitpicking, but the image shows you’re are turning the color from black to red. Please be careful on the presentation.”:
We have shown the grayscale image in this report

3. “Good to mention these all. Please clearly describe how you overcame these difficulties”:
We have mentioned in methods how the project took over from the midterm checkpoint.

4. “Prepare different rules for experiments by increasing the complexity (to make it more interesting)”:
The complexity was increased by considering cases with multiple objects and also cases with overlapping and non-overlapping objects in the images.

5. “In your final report (introduction), please summarize the these works and stress how different your work is from them”:
We have reviewed this in the background section of this report

6.”In this report, how reward function is modeled is not clear. Please make it clear in your final report to answer this question better”:
The reward function for our model is:
 
References:
[1] https://pdfs.semanticscholar.org/febb/285861f40c53f45937c0283d1867f4c6d9bf.pdf
[2] http://cs229.stanford.edu/proj2015/163_report.pdf
[3] http://richzhang.github.io/colorization/
[4] https://opencv-python-tutroals.readthedocs.io/en/latest/












