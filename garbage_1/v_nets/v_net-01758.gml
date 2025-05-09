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
  id 1758
  arrival_time 39150.63450640524
  lifetime 1923.3636087659775
  num_nodes 6
  type "path"
  node [
    id 0
    label "0"
    cpu 13
    gpu 19
    rom 50
  ]
  node [
    id 1
    label "1"
    cpu 36
    gpu 2
    rom 18
  ]
  node [
    id 2
    label "2"
    cpu 10
    gpu 1
    rom 16
  ]
  node [
    id 3
    label "3"
    cpu 41
    gpu 44
    rom 46
  ]
  node [
    id 4
    label "4"
    cpu 6
    gpu 30
    rom 37
  ]
  node [
    id 5
    label "5"
    cpu 43
    gpu 25
    rom 21
  ]
  edge [
    source 0
    target 1
    bw 42
  ]
  edge [
    source 1
    target 2
    bw 45
  ]
  edge [
    source 2
    target 3
    bw 17
  ]
  edge [
    source 3
    target 4
    bw 30
  ]
  edge [
    source 4
    target 5
    bw 45
  ]
]
