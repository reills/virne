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
  id 751
  arrival_time 15922.995946962455
  lifetime 891.9197927236737
  num_nodes 2
  type "path"
  node [
    id 0
    label "0"
    cpu 7
    gpu 44
    rom 32
  ]
  node [
    id 1
    label "1"
    cpu 25
    gpu 37
    rom 7
  ]
  edge [
    source 0
    target 1
    bw 24
  ]
]
