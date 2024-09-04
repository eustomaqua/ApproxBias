# ApproxBias

![CircleCI](https://img.shields.io/circleci/build/github/eustomaqua/ApproxBias/master) 
 

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/eustomaqua/ApproxBias/tree/master.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/eustomaqua/ApproxBias/tree/master) 
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/d0f9d3235ebf4454b3f43beb137bb2c7)](https://app.codacy.com/gh/eustomaqua/ApproxBias/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade) 

Does machine bring in extra bias in learning? Approximating discrimination within models quickly 


### Requirements

```shell
$ # Create a virtual environment
$ conda create -n test python=3.8
$ source activate test

$ # Install packages
$ pip install -r requirements.txt
$ pip install pytest==8.3.2
$ pip install carbontracker==1.2.5

$ # Delete the virtual environment
$ source deactivate
$ conda remove -n test --all
```

### Getting started

```shell
$ # python -m pytest .
```
