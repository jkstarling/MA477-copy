
<h2> ===================================================</h2>                               
 <h1><center>MA477 - Theory and Applications of Data Science</center></h1> 
  <h1><center>Lesson 2: Introduction to NumPy</center></h1> 
 
 <h4><center>Dr. Valmir Bucaj</center></h4>
 <center>United States Military Academy, West Point</center> 
 <center> AY20-2</center>
<h2>===================================================</h2>

NumPy (or Numpy) is the main library for scientific computing in Python. It provides a high-performance multidimensional array object, and tools for working with these arrays. One may think of NumPy as a Linear Algebra library for Python. 

Numpy is super important when doing Data Science and Machine Learning with Python as almost all of the libraries in the PyData Ecosystem rely on Numpy as one of their most important building blocks!

NumPy Python gives functionality comparable to MATLAB and they both allow the user to write fast programs as long as most operations work on arrays or matrices instead of scalars. In comparison, MATLAB boasts a large number of additional toolboxes, notably Simulink, whereas NumPy is intrinsically integrated with Python, a more modern and complete programming language. Moreover, complementary Python packages are available; SciPy is a library that adds more MATLAB-like functionality and Matplotlib is a plotting package that provides MATLAB-like plotting functionality.

Numpy is quite fast as, under the hood, it has direct bindings to C libraries.

To use Numpy you need to first import it into the Notebook. It is customary to import it using ``np`` as an alias:


```python
import numpy as np
```

<h2>NumPy Arrays</h2>
<br>
Numpy arrays are the main way in which we will be using Numpy in this course. We will almost exclusively use only the following two types of arrays:

- *1-D Arrays, also known as vectors*, and 
- *2-D Arrays, also known as matrices.*
    
<b>Remark:</b> there are, however, multidimensional arrays which are used for example when conducting image classification. We will give examples for multidmensional arrays below as well, but we will most likely not encounter them again for the duration of this course. 

Regardless of the dimension of arrays, it is standard to refer to them simply as <b>arrays</b>.

We will cover a few categories of basic array manipulations here:

- <i>Multiple ways of creating numpy arrays</i>
- *Attributes of arrays*: Determining the size, shape, and data types of arrays
- *Indexing of arrays*: Extracting and setting the value of individual array elements
- *Slicing of arrays*: Extracting and setting smaller subarrays within a larger array
- *Reshaping of arrays*: Changing the shape of a given array
- *Joining and splitting of arrays*: Combining multiple arrays into one, and splitting one array into many
<br>
<h2> Creating NumPy Arrays</h2>
<br>

<ol style="list-style-type:square;"> 
    
<li><h3>Creating N-Dimensional NumPy Arrays From Python Objects</h3></li>
     <ol style='list-style-type:circle;'>
         <br>
      <li><b>1-D Arrays From Python Lists</b></li> First, we create a Python list
        
```python
my_list=[1,2,3] 
```
We can cast that list into a Numpy array:

```python
arr=np.array(my_list)
#Result
array([1, 2, 3])
```
<li><b> 2-D Arrays From Lists of Lists</b></li>
First create a list of lists

```python
my2D_mat=[[1,1,1],[2,2,2],[3,3,3],[4,4,4]]
```

we can cast this list of lists into a numpy array to create a 4x3 matrix:

```python
mat2D_arr=np.array(my_mat)

#Result
array([[1, 1, 1],
       [2, 2, 2],
       [3, 3, 3],
       [4, 4, 4]])
```
   <br><li><b> 3-D Arrays From Lists of Lists</b></li>
First we create a *nested* list of lists

```python
my3D_mat=[[[1,1,1],[1,1,1]],[[2,2,2],[2,2,2]],[[3,3,3],[3,3,3]],[[4,4,4],[4,4,4]]]
```
We cast this into a 3-dimensional 4x2x3 array by:
```python
my3D_arr=np.array(my3D_mat)
#Result 
array([[[1, 1, 1],
        [1, 1, 1]],

       [[2, 2, 2],
        [2, 2, 2]],

       [[3, 3, 3],
        [3, 3, 3]],

       [[4, 4, 4],
        [4, 4, 4]]])
```
</ol>
   

<li><h3>Creating N-Dimensional NumPy Arrays From NumPy Built-in Methods</h3></li>

<ol style='list-style-type:circle;'>
<br>
<li> <b>1-D NumPy Arrays From The <i>Arange</i> Method </b></li>
    
```python 
np.arange(Start, Stop, Step_Size)
```
If no step size is specified, the default is 1

If no start point is specified, the default will be 0

Examples:
```python
np.arange(0,10)
#Result
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

np.arange(10)
#Result
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

np.arange(5,20,2)
#Result
array([ 5,  7,  9, 11, 13, 15, 17, 19]
```

    
    
<li> <b>N-Dim Array of Zeros</b></li>
    
```python
    np.zeros(shape=(rows,columns), dtype=[float,int])
```
Examples

```python
np.zeros(4)
#Result
array([0., 0., 0., 0.])
#Shape
np.zeros(4).shape=(4,)

np.zeros((1,4))
#Result
array([[0, 0, 0, 0]])
#Shape
np.zeros((1,4)).shape=(1,4)

np.zeros((3,5))
#Result
array([[0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.]])
#Shape
np.zeros((3,5)).shape=(3,5)

```

<li><b>N-dim Array of Ones</b></li>

```python
np.ones(5)
#Result
array([1., 1., 1., 1., 1.])

np.ones((5,3))
#Result
array([[1., 1., 1.],
       [1., 1., 1.],
       [1., 1., 1.],
       [1., 1., 1.],
       [1., 1., 1.]])
```
    
<li><b>N-dim Arrays via Linspace</b></li>
It returns <i>num</i> evenly spaced numbers over a prespecified interval

```python
np.linspace(start,stop, number of points)
```
Examples:

```python
np.linspace(0,1,5, endpoint=True)
#Result
array([0.  , 0.25, 0.5 , 0.75, 1.  ])

np.linspace(0,1,5,endpoint=False)
#Result
array([0. , 0.2, 0.4, 0.6, 0.8])
```

<li><b>Identity Matrix</b></li>

```python
np.eye(num_rows,num_columns,k=choice of diagonal)
    ```
 Examples:
 
 ```python
np.eye(5)
#Result
array([[1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.]])

np.eye(5,k=1)
#Result
array([[0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.],
       [0., 0., 0., 0., 0.]])

np.eye(5,k=-1)
#Result
array([[0., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.]])

np.eye(4,7,dtype=int)
#Result
array([[1, 0, 0, 0, 0, 0, 0],
       [0, 1, 0, 0, 0, 0, 0],
       [0, 0, 1, 0, 0, 0, 0],
       [0, 0, 0, 1, 0, 0, 0]])

```
    
    
   </ol>
<li><h3>Creating N-Dimensional NumPy Arrays of Random Numbers</h3></li>
<br>

<ol>
    <li> <b>Sampling from a uniform distribution</b>
        
 ```python
   np.random.rand(dimensions/shape of the array=?)
    ```
   Example:
   
   ```python
np.random.rand(8)
#Result
array([0.27941405, 0.85130713, 0.88800973, 0.99857744, 0.38542876,
       0.1056563 , 0.44328663, 0.53735746])

np.random.rand(3,4)
#Result
array([[0.16771053, 0.31255401, 0.73049355, 0.80661161],
       [0.79484696, 0.29320526, 0.61743941, 0.34803172],
       [0.97730059, 0.06073042, 0.85747032, 0.26421474]])

np.random.rand(3,2,6)
#Result
array([[[0.16438327, 0.59142023, 0.42819976, 0.67578927, 0.21987979],
        [0.17355946, 0.78705391, 0.03286594, 0.10033335, 0.19749326]],

       [[0.34572113, 0.33834082, 0.12847944, 0.87190941, 0.60720626],
        [0.55263701, 0.10490326, 0.43979857, 0.56746772, 0.6107678 ]],

       [[0.21584038, 0.87023275, 0.28063209, 0.57454712, 0.31576353],
        [0.15555137, 0.059627  , 0.1543733 , 0.10180803, 0.11313844]]])

```

<li> <b>Sampling from a standard-normal distribution</b></li>

 ```python
   np.random.randn(dimensions/shape of the array=?)
    ```
Examples:

```python
np.random.randn(5)
#Result
array([ 1.26623342,  1.17820516,  0.52915335, -1.01814125, -3.41636585])

np.random.randn(3,5)
#Result
array([[-0.99265674, -0.45074712,  1.46365884, -0.36128292,  1.3162904 ],
       [ 0.35586714, -0.7646797 , -0.13169354, -0.134049  , -0.49413122],
       [ 0.06557251, -0.78137674, -0.99484843,  0.32590831, -1.63439904]])
```

<li> <b>Sampling random integers</b></li>

```python
np.random.randint(low(inclusive),high(exclusive), size=(?,?))
```
By default size=1

Examples:

```python
np.random.randn(200)
#Result
A random number between 0 and 199

np.random.randint(low=2,high=7,size=9)
#Result
array([6, 2, 4, 6, 3, 5, 6, 2, 6])

np.random.randint(2,7,size=(3,5))
#Result
array([[6, 2, 2, 2, 6],
       [6, 3, 6, 3, 2],
       [2, 2, 3, 4, 3]])
```

</ol>

</ol>


```python

```

<h2><b>Elementary Numpy Methods and Operations</b></h2>
<ol style="list-style-type:square;">

<li><b> Reshaping Numpy Arrays</b></li>
Often it will be important to reshape a given array by specifying the desired dimensions. 

Suppose we are given the following array:

```python
arr=np.random.randint(0,100,20)
#Result
array([40, 94, 15, 79, 38, 23, 21, 32, 20, 61, 41, 68, 24, 21, 50, 65, 57,
       72,  7, 83])

#Reshaping

new_arr=arr.reshape(4,5)
#Result
array([[40, 94, 15, 79, 38],
       [23, 21, 32, 20, 61],
       [41, 68, 24, 21, 50],
       [65, 57, 72,  7, 83]])

arr.reshape(2,10)
#Result
array([[40, 94, 15, 79, 38, 23, 21, 32, 20, 61],
       [41, 68, 24, 21, 50, 65, 57, 72,  7, 83]])
```

<li><b>Maximums and Minimums </b></li>

Often it is important to know what is the largest values in an array as well as finding the index
where they are located in the array.

Using the random array above we have:


```python
arr.max()
#Result
94

arr.min()
#Result 
7

arr.argmax()
#Result
1

arr.argmin()
#Result
18

new_arr.max(axis=0)
#Result
array([65, 94, 72, 79, 83])

new_arr.max(axis=1)
#Result
array([94, 61, 68, 83])

new_arr.argmax(axis=1)
#Result
array([1, 4, 1, 4], dtype=int64)
```

<li><b>Data Type </b></li>

When building prediction and classification models you want to make sure that you are feeding
in the correct data type into the models. So, it becomes important to know how to check the data type
of the arrays at hand.

```python
arr.dtype
#Result
dtype('int32')
```



    
    
</ol>


<h2><b>Array Indexing and Selection</b></h2>
<ol style="list-style-type:square;">

<li><b> Indexing of 1D Numpy Arrays</b></li>

Selecting elements from a Numpy array is very similar to selecting elements from a Python list, via square
brackets and slice notation.

Example: 
```python
arr=np.arange(5,33,3)
#Result
array([ 5,  8, 11, 14, 17, 20, 23, 26, 29, 32])
```
If we want to pick out an element at a certain position we can do so via square brackets(remember indexing starts at 0). Specifically, if we want to pick out the third element from the array ``arr`` above we can do so via:

```python
arr[2]
#Result
11
```

If we want to select values in a certain range, we can do so by using slice notation:

```python
arr[start_index(inclusive):end_index(exlusive)]

arr[1:6]
#Result
array([ 8, 11, 14, 17, 20])
```

If we want o select values from the first element up to a stoping point we don't need to specify the starting position:

```python
arr[:6]
#Result
array([ 5,  8, 11, 14, 17, 20])
```

Similarly, from a fixed position all the way to the last element:

```python
arr[5:]
#Result
array([20, 23, 26, 29, 32])
```

Picking out the last element:

```python
arr[-1]
#Result
32
```
Picking a range of elements starting from the last one:

```python
arr[-3:]
#Result
array([26, 29, 32])
```

<li><b> Indexing of 2D Numpy Arrays</b></li>
Suppose we are given the following 2D numpy array:

```python
arr_2d=np.array([[22, 27,  6,  8, 17],
                   [10, 15,  3,  9,  3],
                   [ 2,  9, 14, 13,  4]])
```

If we want to select the element 4 in the lower right corner we can do so in two ways:

```python 
arr_2d[2,4]
arr_2d[2][4]
```
If we wanted to select the following 2D subarrays:

```python
array([[ 3,  9,  3],
       [14, 13,  4]])

array([[ 8, 17],
       [ 9,  3],
       [13,  4]])
```
We can do so as follows:

```python
arr_2d[1:,2:]

arr_2d[:,3:]

```
<li><b><font size='3'> Conditional Selection</font></b></li>
 <br>
 Often we want to select only elements of an array that satisfy a certain condition. 
 
 For example, given the array below
```python
arr=np.arange(1,15)
array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])
```
Suppose we only want to select elements that are larger than 5. We can do that as follows

```python
arr[arr>5]
#Result
array([ 6,  7,  8,  9, 10, 11, 12, 13, 14])
```
Or, suppose we want to select only even elements:

```python
arr[arr%2==0]
#Result
array([ 2,  4,  6,  8, 10, 12, 14])
```

Or, elemnets that are odd and smaller than 10:

```python
arr[(arr%2!=0) & (arr<10)]
#Result
array([1, 3, 5, 7, 9])
```
</ol>


```python

```

<h2><b>More on Numpy Operations and Universal Array Functions</b></h2>
<ol style="list-style-type:square;">

<li><b> Array with Array Operations</b></li>

All elementary operations between two numpy arrays are performed on an element-to-element basis. 

For example let's take the following two arrays:

```python
arr1=np.arange(1,10)=array([1,  2,  3,  4,  5,  6,  7,  8,  9, 10])

arr2=np.arange(11,21)=array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
```
Addition, Subtraction, Multiplication, Division, Exponentiation:

```python
arr2+arr1=array([12, 14, 16, 18, 20, 22, 24, 26, 28, 30])

arr2-arr1=array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10])

arr2*arr1=array([ 11,  24,  39,  56,  75,  96, 119, 144, 171, 200])

arr2/arr1=array([11., 6., 4.33, 3.5 , 3., 2.67, 2.43, 2.25, 2.11, 2. ])

arr1**2=array([  1,   4,   9,  16,  25,  36,  49,  64,  81, 100], dtype=int32)

arr1**arr1=array([ 1, 4, 27, 256, 3125, 46656, 823543, 16777216, 387420489, 1410065408], dtype=int32)
```

<li><b><font size='4'> Arrays with Scalars & Broadcasting</font></b></li>

Unlike adding a number to a vector, it is possible to **add, subtract, multiply, divide**, a scalar to a numpy array. 
In Numpy, the way this is acomplished is by **broadcasting** that scalar to every number in the array, so in essence,
the operation is also done element-wise.

For example let ``arr1`` be as above, then 

```python
arr1+4=array([ 5,  6,  7,  8,  9, 10, 11, 12, 13, 14])

arr1-4=array([-3, -2, -1,  0,  1,  2,  3,  4,  5,  6])

arr1*4=array([ 4,  8, 12, 16, 20, 24, 28, 32, 36, 40]

arr1/4=array([0.25, 0.5 , 0.75, 1.  , 1.25, 1.5 , 1.75, 2.  , 2.25, 2.5 ])  
```
Essentially what happens *under the hood* is that numpy first creates an array of the same shape as ``arr1`` consisting of only the number ``4`` and then performs the operation on the two arrays.

That is, numpy will first create an array

```python
arr_of_fours=np.full_like(arr1,4)
            =array([4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
```
and then carries out the element-wise operation between ``arr1`` and ``arr_of_fours``.

<li><b><font size='3'> Universal Array Functions</font></b></li>

Universal Array Functions (short uafs) are simply mathematical functions/operations that you can apply on an array.

Such functions include:

<b> Square Root</b>
```python 
    np.sqrt(arr1)=array([1., 1.414, 1.732, 2., 2.236, 2.449, 2.646, 2.828, 3. ,3.162])
```

<b>Exponentiating </b>
```python
np.exp(arr1)=array([2.71828183e+00, 7.38905610e+00, 2.00855369e+01, 5.45981500e+01,
       1.48413159e+02, 4.03428793e+02, 1.09663316e+03, 2.98095799e+03,
       8.10308393e+03, 2.20264658e+04])
```
<b>Trig Functions</b>
```python
np.sin(arr1)=array([ 0.841, 0.909, 0.141, -0.757, -0.959, -0.279, 0.657, 0.989, 0.412, -0.544])

np.cos(arr1)=array([ 0.54 , -0.416, -0.99 , -0.654,  0.284,  0.96 ,  0.754, -0.146, -0.911, -0.839])
```

<b> Logarithmic Functions</b>
```python
np.log10(arr1)=array([0., 0.301, 0.477, 0.602, 0.699, 0.778, 0.845, 0.903, 0.954, 1.])
```

<li><b><format size='4'>Aggregation Functions</format></b></li>

NumPy provides many other aggregation functions, but we won't discuss them in detail here.
Additionally, most aggregates have a ``NaN``-safe counterpart that computes the result while ignoring missing values.


The following table provides a list of useful aggregation functions available in NumPy:

|Function Name      |   NaN-safe Version  | Description                                   |
|-------------------|---------------------|-----------------------------------------------|
| ``np.sum``        | ``np.nansum``       | Compute sum of elements                       |
| ``np.prod``       | ``np.nanprod``      | Compute product of elements                   |
| ``np.mean``       | ``np.nanmean``      | Compute mean of elements                      |
| ``np.std``        | ``np.nanstd``       | Compute standard deviation                    |
| ``np.var``        | ``np.nanvar``       | Compute variance                              |
| ``np.min``        | ``np.nanmin``       | Find minimum value                            |
| ``np.max``        | ``np.nanmax``       | Find maximum value                            |
| ``np.argmin``     | ``np.nanargmin``    | Find index of minimum value                   |
| ``np.argmax``     | ``np.nanargmax``    | Find index of maximum value                   |
| ``np.median``     | ``np.nanmedian``    | Compute median of elements                    |
| ``np.percentile`` | ``np.nanpercentile``| Compute rank-based statistics of elements     |
| ``np.any``        | N/A                 | Evaluate whether any elements are true        |
| ``np.all``        | N/A                 | Evaluate whether all elements are true        |


For a more extensive discussion you may visit the official documentation page for Numpy ufuncts: __[Numpyu Universal Functions](https://docs.scipy.org/doc/numpy/reference/ufuncs.html#ufunc)__

</ol>



<h2>====================================================</h2>

<h1>HOMEWORK #1</h1>

<h2>====================================================</h2>

<h3> Complete the following exercises</h3>
<br>

<ol>
<li><font size=3> Create a 5 by 7 numpy array of random integers with values between 3 and 11 inclusive. Then do the following:
    <ol> <li>Check the data type of the array</li>
        <li>Find the maximum of the entire array</li>
        <li>Find the minimum value of each row</li>
        <li>Find the index where the minimum value of each row occurs</li>
    </ol></font></li>
</ol>


```python
#Your code for part A of Exercise 1 goes here

```


```python
#Your code for part B of Exercise 1 goes here

```


```python
#Your code for part C of Exercise 1 goes here

```


```python
#Your code for part D of Exercise 1 goes here

```

<ol start='2'>
   
<li> <font size='3'>Enter the code below that generates the following numpy array: </font></li>

```python
#Result
array([ 4,  9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84,
       89, 94, 99])
```    
</ol>


```python
#Your code for Exercise 2 goes here

```

<ol start='3'>
   
<li> <font size='3'>Enter the code below that generates the following numpy array: </font></li>

```python
#Result
array([[ 4,  9, 14, 19, 24],
       [29, 34, 39, 44, 49],
       [54, 59, 64, 69, 74],
       [79, 84, 89, 94, 99]])
```    
</ol>


```python
#Your code for Exercise 3 goes here

```

<ol start='4'>
   
<li> <font size='3'>Create a 4 by 2 by 3 array of normally distributed values</font></li>
   
</ol>


```python
#Your code for Exercise 4 goes here

```

<ol start='5'>
   
<li> <font size='3'>Enter the code below that generates the following numpy array(including the correct
    data type): </font></li>

```python
#Result
array([[0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [1, 0, 0, 0, 0, 0],
       [0, 1, 0, 0, 0, 0],
       [0, 0, 1, 0, 0, 0],
       [0, 0, 0, 1, 0, 0]]
```    

</ol>


```python
#Your code for Exercise 5 goes here

```

<ol start='6'>
   
<li> <font size='3'>Given the numpy array below </font></li>

```python
array([ 3,  5,  7,  9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33])
```    
<ol>
    <li>Enter the code below that selects only the last four elements:</li>

```python
#Result
array([27, 29, 31, 33])
```
<li>Enter the code that selects the following subarray:

```python
#Result
array([ 3,  9, 15, 21, 27, 33])
```
</ol>
</ol>


```python
#Your code for part A of Exercise 6 goes here

```


```python
#Your code for part B of Exercise 6 goes here

```

<ol start='7'>
   
<li> <font size='3'>Given the 2D numpy array below </font></li>

```python
arr_2d=np.array([[69, 75, 52, 33,  7, 62, 22],
               [87, 51, 13, 75, 78, 46, 53],
               [43, 24, 58, 76,  3,  3, 57],
               [84, 86, 75, 26, 71, 77,  7],
               [75, 17, 17, 38, 54, 72, 78]])
```    
<ol>
    <li> Enter the code below that selects the element 72 on the last row</li>
    <br>
   <li>Enter the code below that selects the following subarray:</li>

```python
#Result
array([[76,  3],
       [26, 71],
       [38, 54]])
```
<li>Enter the code that selects the following subarray:

```python
#Result
array([[87, 51, 13, 75, 78, 46, 53],
       [43, 24, 58, 76,  3,  3, 57],
       [84, 86, 75, 26, 71, 77,  7]])
```
</ol>
</ol>


```python
#Your code for part A of Exercise 7 goes here

```


```python
#Your code for part B of Exercise 7 goes here

```

<ol start='8'>
   
<li> <font size='3'>Given the numpy array below </font></li>

```python
arr=np.arange(1,50,3)
array([ 1,  4,  7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49])
```    
<ol>
    <li>Enter the code that selects all elements larger than 7 and smaller than 43:

```python
#Result
array([10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40])

```
   <li>Enter the code below that selects all even integers larger than 11 and smaller than 45:</li>

```python
#Result
array([16, 22, 28, 34, 40])
```
</ol>
</ol>


```python
#Your code for part A of Exercise 8 goes here

```


```python
#Your code for part B of Exercise 8 goes here

```

<ol start='9'>
   
<li> <font size='3'>Create a 1D uniformly distributed array with 20 elements and compute: </font></li>


<ol>
    <li> Mean</li>

   <li>Standard Deviation</li>
   
   <li>Median</li>
   
   <li>Mean of its log values</li>
</ol>
</ol>


```python
#Your code for part A of Exercise 9 goes here

```


```python
#Your code for part B of Exercise 9 goes here

```


```python
#Your code for part C of Exercise 9 goes here

```


```python
#Your code for part D of Exercise 9 goes here

```

<ol start='10'>
   
<li> <font size='3'>Write a function that:</font></li>
    <ol>
    <br>
   <li> takes a 1D Numpy array as input and prints out the <b>reversed</b> array with element type float</li>

For example, if you input the following array into your function
```python
array([ 1,  3,  5,  7,  9, 11, 13, 15, 17, 19, 21, 23])
```
your function should print out

```python
array([23., 21., 19., 17., 15., 13., 11.,  9.,  7.,  5.,  3.,  1.])
```

<li> takes a 2D Numpy array as input and prints out a copy of the input array <b> flattened</b> to one dimension</li>

For example if the input array is:
```python
array([[7, 4, 8, 3],
       [6, 3, 4, 3],
       [4, 7, 5, 4]])
```
the output of your function should be
```python
array([7, 4, 8, 3, 6, 3, 4, 3, 4, 7, 5, 4])
```
</ol>
</ol>


```python
#Your code for part A of Exercise 10 goes here

```


```python
#Your code for part A of Exercise 10 goes here

```
