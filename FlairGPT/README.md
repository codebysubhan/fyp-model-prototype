# FlairGPT: Functional Layouts for Aesthetic Interior Realisations 

The code for *FlairGPT: Functional Layouts for Aesthetic Interior Realisations*  
Links:  
- [Paper](https://arxiv.org/abs/2501.04648)  
- [Webpage](https://flairgpt.github.io/)  

# Requirements 
Install the requirements
```bash

cd Scene_Synthesis
conda create -n flairgpt
conda activate flairgpt
conda install pip
pip install -r requirements.txt
```

# Create the 'hidden.env' file 
```env 
OPENAI_API_KEY = "YOUR_OPENAI_KEY"
```

## Run
There are two options for running the code: 
1. run_scene_synthesis.ipynb  
2. scene_synthesis.py  

We recommend using the notebook (run_scene_synthesis.ipynb) if you would like to see outputs of constraints/re-run cells that fail due to LLM errors (hallucinations etc), or to rerun optimisations that give suboptimal results. Otherwise, run the .py file. 

For both options, please edit scene_descriptor to be your desired room description.  
The output can be found in Scene_Synthesis/Result_txt. Please note that object retrieval is not implemented in this work. 

## Example Results
![gallery](gallery.png)
