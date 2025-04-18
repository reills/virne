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
  id 54
  arrival_time 1138.0792168061032
  lifetime 175.36903639494366
  num_nodes 6
  type "path"
  node [
    id 0
    label "0"
    cpu 22
    gpu 19
    rom 48
  ]
  node [
    id 1
    label "1"
    cpu 43
    gpu 50
    rom 1
  ]
  node [
    id 2
    label "2"
    cpu 49
    gpu 23
    rom 3
  ]
  node [
    id 3
    label "3"
    cpu 15
    gpu 12
    rom 36
  ]
  node [
    id 4
    label "4"
    cpu 6
    gpu 43
    rom 39
  ]
  node [
    id 5
    label "5"
    cpu 43
    gpu 50
    rom 10
  ]
  edge [
    source 0
    target 1
    bw 46
  ]
  edge [
    source 1
    target 2
    bw 15
  ]
  edge [
    source 2
    target 3
    bw 22
  ]
  edge [
    source 3
    target 4
    bw 20
  ]
  edge [
    source 4
    target 5
    bw 26
  ]
]
