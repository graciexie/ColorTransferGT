# Optimal Transport Research

Supervising Professor: Dr. Xiaoming Huo

PhD Students: Yiling Xie, Yiling Luo

Undergrad Student: Samuel Hart

Repository to store our code and report for a research project in Optimal Transport.

To run the code in the AlgoCode folder, one should clone the repository onto their machine. It is recommended that you create a virtual environment to run the code rather than using the global environment on your machine. If you wish to create a virtual environement, see the relevant links below.

Using Python:
https://docs.python.org/3/library/venv.html

Using Anaconda:
https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/

The requirements/dependencies are stored in the requirements.txt file. These dependencies are NumPy and Matplotlib. These dependencies can be installed using the commands found in the links above. Note if you are not using a virtual environment or have not activated your virtual environment, then these will be installed to your global Python or Conda environment.

Now, the Color Transfer experiment can be run with the following command:

python main.py

At the time of writing this, this should return the results of Color Transfer using the Sinkhorn, Greenkhorn, Stochastic Sinkhorn, SAG, APDAGD, and AAM. Note that the results appear in the results directory within the AlgoCode directory.

There is also a basic second test located in the test.py file that can be run with the following command:

python test.py

This uses each algorithm to solve the Optimal Transport problem with a random source and target measure, as well as a cost matrix defined by the Euclidean distance.

For a full report on this repository check out the ResearchReport PDF!
