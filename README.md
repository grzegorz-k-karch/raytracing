@startuml

@enduml

Architecture:
  - building blocks 
    (should cover items in requirements, decoupled from other blocks, one area of responsibility)
	- command-line parser
	- scene parser - should accept scene description and produce list of objects -> objects
	- objects - classes containing objects <- list of objects from scene parser
	- renderer - takes command line arguments and renders objects <- objects, render params; -> rendered image
	- file output manager - saves the rendered image <- rendered image
  - major classes 
    (their responsibilities, interactions with other classes, class hierarchies, state transitions, alternative class design)
	- object
	  - math object
	  - triangle-mesh object
	- material
	- arg parser
	- scene parser
	- renderer
	  - CUDA renderer
	- image writer
	  - ppm writer
	  - jpeg writer
  - data design
  - user interface design
    - command line, config files
  - input/output
  - erro processing
  - change strategy
    - changing from CUDA to Vulkan
