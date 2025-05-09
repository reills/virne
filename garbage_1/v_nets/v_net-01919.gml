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
  id 1919
  arrival_time 42107.93161159907
  lifetime 217.03603894865233
  num_nodes 4
  type "path"
  node [
    id 0
    label "0"
    cpu 8
    gpu 15
    rom 24
  ]
  node [
    id 1
    label "1"
    cpu 37
    gpu 34
    rom 1
  ]
  node [
    id 2
    label "2"
    cpu 4
    gpu 4
    rom 17
  ]
  node [
    id 3
    label "3"
    cpu 25
    gpu 24
    rom 0
  ]
  edge [
    source 0
    target 1
    bw 28
  ]
  edge [
    source 1
    target 2
    bw 9
  ]
  edge [
    source 2
    target 3
    bw 46
  ]
]
