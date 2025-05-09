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
  id 772
  arrival_time 16150.084119439543
  lifetime 1850.5490036582107
  num_nodes 7
  type "path"
  node [
    id 0
    label "0"
    cpu 11
    gpu 17
    rom 16
  ]
  node [
    id 1
    label "1"
    cpu 7
    gpu 28
    rom 4
  ]
  node [
    id 2
    label "2"
    cpu 14
    gpu 6
    rom 6
  ]
  node [
    id 3
    label "3"
    cpu 20
    gpu 14
    rom 20
  ]
  node [
    id 4
    label "4"
    cpu 30
    gpu 14
    rom 42
  ]
  node [
    id 5
    label "5"
    cpu 13
    gpu 14
    rom 43
  ]
  node [
    id 6
    label "6"
    cpu 29
    gpu 14
    rom 25
  ]
  edge [
    source 0
    target 1
    bw 19
  ]
  edge [
    source 1
    target 2
    bw 24
  ]
  edge [
    source 2
    target 3
    bw 30
  ]
  edge [
    source 3
    target 4
    bw 47
  ]
  edge [
    source 4
    target 5
    bw 40
  ]
  edge [
    source 5
    target 6
    bw 21
  ]
]
