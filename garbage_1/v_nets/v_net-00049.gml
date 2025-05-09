graph [
  node_attrs_setting [
    name "cpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "gpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "rom"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  link_attrs_setting "_networkx_list_start"
  link_attrs_setting [
    name "bw"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "link"
    type "resource"
  ]
  id 49
  arrival_time 987.0090367673613
  lifetime 450.31759159223293
  num_nodes 3
  type "path"
  node [
    id 0
    label "0"
    cpu 36
    gpu 43
    rom 0
  ]
  node [
    id 1
    label "1"
    cpu 3
    gpu 5
    rom 24
  ]
  node [
    id 2
    label "2"
    cpu 46
    gpu 12
    rom 16
  ]
  edge [
    source 0
    target 1
    bw 31
  ]
  edge [
    source 1
    target 2
    bw 25
  ]
]
