# AI kidney diagnosis assistant
Master thesis project at the department of computer science at NTNU.

## Run the ML code using conda
The machine learning directory contains an environment.yml file. This file can be used to build a conda virtual environment to obtain the dependencies needed to run the project. By using the command `conda env create -f environment.yml` a first time user can obtain all of the dependencies in the AI-diagnostic virtual environment. To set the python path for the project run `python setup.py`. Activate the environment using `conda activate AI-diagnostic`. Use `which python` to ensure that the correct python is in use. Python code can now be run from the terminal. To deactivate the environment, use `conda deactivate`. The environment can be deleted by using `conda remove -n AI-diagnostic --all` which will remove all of the dependencies and delete the environment. After the environment is created, packages will not be updated by running the create command again. Use conda to install the needed ones, or delete and rerun the create command using the environment.yml file.

## Dataset
The repository does not contain the data. Download the data from the [Database of dynamic renal scintigraphy](https://dynamicrenalstudy.org/) and place it inside a directory named data in the root level of the repository. It will be ignored by git. Then unpack the zip files for drsprg and drsbru inside the BAZA dynamicrenal directory. Find the drsprg.csv file with labels inside the drsprg directory under BAZA dynamicrenal and place it in a new directory called labels under data. Create a csv file with labels for drsbru on the same format and place it together with the drsprg.csv file.

The data directory should now include this:

AI_kidney_diagnosis_assistant/
                            data/
                                BAZA dynamicrenal/
                                            drsbru/
                                            drsprg/
                                labels/
                                    drsbru.csv
                                    drsprg.csv
                                segmentation_masks/
                                            drsbru/
                                            drsprg/


Run the construct_data.py script in the ML directory to create the dataset from these files. You only need to do this once.
