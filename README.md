# Generic ML Experimentation Workflow
I am tired of rewriting ML experimentation pipelines 40! times so this is a generic template that can be used to start new research projects. 
## Requirements
	idk lol 2
## How 2 Use
	Edit/write in modules defined below and add/remove modules as necessary 		     
	to fit project. Make sure to also update each [__init__.py] file so 
	module functions can be invoked from other files.
	
	Define run args in [main.py] and update [run.sh] accordingly.
	run [run.sh] with desired parameters to start a run 
	(however you decide to define "run").

	Use [scratchpad.ipynb] for quick and dirty debugging.
## Modules
### data_process 
	Contains scripts/files to load and process datasets 
### datasets
	Put your datasets here. Files in this folder will not be git tracked 
	by default
### logger_runs
	Logged training/evaluation runs will be saved to this directory. 
	Files in this folder will not be git tracked by default.
### models
	Contains model declaration classes
###  train_evaluate
	Contains train/evaluate scripts
### utility
	Whatever extra stuff you want here. Currently contains the [run.py]
	script to run an experiment with the desired args

### notification
	simple email forwarding for update notifications, [email_args.py] has been omitted from repo for security reasons.
	To use this module, create the file and put the following variables in these exact names with the requisite info:
	
	SOURCE_EMAIL = 'sender@email.com'
	PASSWORD = 'dummypw'
	DESTINATION_EMAIL = 'reciever@email.com'
	HEADER = 'dummy header'
	SUBJECT = 'dummy subject'
	
