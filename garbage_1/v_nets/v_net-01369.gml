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
  id 1369
  arrival_time 29105.112977976103
  lifetime 1451.6236312839685
  num_nodes 4
  type "path"
  node [
    id 0
    label "0"
    cpu 36
    gpu 48
    rom 35
  ]
  node [
    id 1
    label "1"
    cpu 15
    gpu 7
    rom 15
  ]
  node [
    id 2
    label "2"
    cpu 27
    gpu 19
    rom 37
  ]
  node [
    id 3
    label "3"
    cpu 46
    gpu 43
    rom 3
  ]
  edge [
    source 0
    target 1
    bw 1
  ]
  edge [
    source 1
    target 2
    bw 13
  ]
  edge [
    source 2
    target 3
    bw 23
  ]
]
