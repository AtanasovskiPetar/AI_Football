<h1>Cloud-trained custom open ai gymnasium environment for a 3v3 python football game.</h1>

<h1>Custom OpenAI Gymnasium Environment</h1>
<p>Created my Custom OpenAI Gymnasium Environment for a 3v3 python football game called AiFootball-v0</p>
<h1>Stable Baselines 3 ML Models</h1>
<p>For training my environment agent I used stable baselines PPO model</p>
<h1>Training the model on Azure Cloud</h1>
<p>I created an Azure account where I created compute cluster and used ssh to connect to my compute cluster where I set up the virtual environment and managed to run my training script.</p>
<br>
<h1>How To Run The Project</h1>
<ol>
<li>ssh -i key_path azureuser@azure_ir -p port</li>
<li>conda deactivate</li>
<li>sudo apt update</li>
<li>sudo apt install python3-venv</li>
<li>python3 -m venv venv</li>
<li>source venv/bin/activate</li>
<li>pip install torch</li>
<li>pip install gymnasium</li>
<li>pip install pygame</li>
<li>pip install stable-baselines3</li>
<li>pip install tensorboard</li>
<li>pip install -e AI_Football/gym-envs</li>
</ol>