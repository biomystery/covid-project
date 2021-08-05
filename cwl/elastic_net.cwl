#!/usr/bin/env cwl-runner

# (Re)generated by BlueBee Platform

$namespaces:
  bb: http://bluebee.com/cwl/
  ilmn-tes: http://platform.illumina.com/rdf/iap/
cwlVersion: cwl:v1.0
class: CommandLineTool 
bb:toolVersion: '1'
label: BS elastic-net model
doc: An elastic-net Logistic regression combined with bootstrapping tool to perform feature selection on binary labeled data. 
hints:
  - class: ResourceRequirement
    ilmn-tes:resources:
      size: small
      type: standard
requirements: 
  - class: DockerRequirement
    dockerPull: 389397373118.dkr.ecr.us-west-1.amazonaws.com/py3-ml:v0.0.1
  - class: InlineJavascriptRequirement

# python run_elasticnet_bootstrap.py -ix test/X.csv -iy test/y.csv -nb 10
baseCommand: ["python", "/py3-ml/run_elasticnet_bootstrap.py"]

inputs: 
  input_expression_matrix: 
    type: File
    inputBinding:
      prefix: -ix
  input_label: 
    type: File
    inputBinding:
      prefix: -iy
  number_of_bootstrapping: 
    type: int?
    default: 100 
    inputBinding:
      prefix: -nb 

outputs:
  results_directory: 
    type: Directory
    outputBinding:
      glob: 'results'