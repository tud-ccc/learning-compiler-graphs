Requirements
--
This artifact is designed to be pretty self-contained and produces all its requirements. In principle it only needs a working docker installation.
See <https://docs.docker.com/install/>

Quick Result Overview
--

It is possible to see the Jupyter notebook without actually running anything. For this, you need a working Jupyter installation (see <https://jupyter.org/install>).

***
Note that the docker container builds Jupyter, so you don't need a local installation if you run the standard method.
***

For this, just execute:
```
jupyter notebook artifact.ipynb
```

Running the artifact
--
This artifact is prepared with a docker container, which will fetch all dependencies, build and prepare the artifact such that it can be executed from the Jupyter notebook `artifact.ipynb`. Make sure you have docker installed and execute the runscript to build and run everything:

```
./run.sh
```

After fetching and compiling everything (warning: this will take several hours), you should see the Jupyter notebook running. It will be forwarded to port 8888. You should have a link with a token in the terminal output, like: 
 
 ```
 https://127.0.0.1:8888/notebooks/artifact.ipynb?token=<some-long-hash-here>
 ```

This link should work on your local browser to see the Jupyter notebook. 

Re-training the models
--
Since training time would take several weeks on a commodity desktop machine, we've included trained models to reproduce our results. If you want to re-train, just run `rm -fr results` to remove the trained models.
