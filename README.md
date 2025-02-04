# 16825-L3D

## Submit code to Canvas
For every assignment you should create a main.py that can be used to run all your code for the assignment, and a README.md file that contains all required documentation. Place all source code used to generate your results, as well as any documentation required to use the code, in a folder named andrewid_code_projX where X is the hw number. Zip the whole folder and submit the zip to Canvas. Here is an example of what your folder structure should look like:

```
andrewid_code_proj1/
    main.py
    README.md
    utils.py
    ....
## zip the whole folder to andrewid_code_proj1.zip:
```

## Submit your webpage to the class website

We will use [Andrew File System(AFS)](https://www.cmu.edu/computing/services/comm-collab/collaboration/afs/how-to/index.html) to store and display webpages. Here is a step by step tutorial for how to do this:

0. Connect to CMU VPN

1. Please make sure to remove all version control directories (e.g. .git) from your website folder, and make sure that the total size is reasonable (less than 20 mb)

2. Place your website under folder projX and zip it. Please make sure that your main report page is called index.html so browsers open it automatically. <br> X is the hw number

3. Remote Copy. Use WinSCP or your favorite scp/ftp tool to copy all your files to your Andrew home directory 
    ```
    scp projX.zip liweiy@linux.andrew.cmu.edu:/afs/andrew.cmu.edu/usr/liweiy/
    ```
4. Log in to a Unix Andrew machine: 
    ```
    ssh liweiy@linux.andrew.cmu.edu
    ```
5. File Transfer. Unzip your website and copy the folder to your project directory:
    ```
    unzip projX.zip -d projX 
    cp -r projX/ /afs/andrew.cmu.edu/course/16/825/www/projects/liweiy/
    ```
   
    The folder structure should look like this:
    ```
    # suppose you are at /afs/andrew.cmu.edu/course/16/825/www/projects/liweiy
    proj1/
        index.html
        data/...
    proj2/
        index.html
        data/...
    ```
6. Publish. The course website needs to be refreshed with your updated files. <br>Do that by going [here](https://www.andrew.cmu.edu/server/publish.html), choosing web pages for a course, and inputing *16-825*.

7. Last step, test your page by visiting: http://www.andrew.cmu.edu/course/16-825/projects/liweiy/projX/
