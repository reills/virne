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
  id 648
  arrival_time 13678.937117774909
  lifetime 98.66428016435891
  num_nodes 4
  type "path"
  node [
    id 0
    label "0"
    cpu 19
    gpu 7
    rom 11
  ]
  node [
    id 1
    label "1"
    cpu 42
    gpu 36
    rom 22
  ]
  node [
    id 2
    label "2"
    cpu 47
    gpu 7
    rom 47
  ]
  node [
    id 3
    label "3"
    cpu 43
    gpu 22
    rom 13
  ]
  edge [
    source 0
    target 1
    bw 9
  ]
  edge [
    source 1
    target 2
    bw 50
  ]
  edge [
    source 2
    target 3
    bw 42
  ]
]
