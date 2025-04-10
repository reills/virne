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
  id 1595
  arrival_time 35844.05119616483
  lifetime 345.26055166077066
  num_nodes 4
  type "path"
  node [
    id 0
    label "0"
    cpu 23
    gpu 9
    rom 49
  ]
  node [
    id 1
    label "1"
    cpu 21
    gpu 50
    rom 25
  ]
  node [
    id 2
    label "2"
    cpu 43
    gpu 35
    rom 32
  ]
  node [
    id 3
    label "3"
    cpu 47
    gpu 14
    rom 46
  ]
  edge [
    source 0
    target 1
    bw 4
  ]
  edge [
    source 1
    target 2
    bw 32
  ]
  edge [
    source 2
    target 3
    bw 48
  ]
]
