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
  id 500
  arrival_time 9431.134848223539
  lifetime 125.61939240096318
  num_nodes 4
  type "path"
  node [
    id 0
    label "0"
    cpu 40
    gpu 18
    rom 41
  ]
  node [
    id 1
    label "1"
    cpu 8
    gpu 40
    rom 4
  ]
  node [
    id 2
    label "2"
    cpu 10
    gpu 19
    rom 34
  ]
  node [
    id 3
    label "3"
    cpu 47
    gpu 12
    rom 32
  ]
  edge [
    source 0
    target 1
    bw 13
  ]
  edge [
    source 1
    target 2
    bw 1
  ]
  edge [
    source 2
    target 3
    bw 23
  ]
]
