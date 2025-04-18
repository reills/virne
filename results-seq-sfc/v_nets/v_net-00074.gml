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
  id 74
  arrival_time 1559.1415941001524
  lifetime 1658.2322905500891
  num_nodes 10
  type "path"
  node [
    id 0
    label "0"
    cpu 21
    gpu 19
    rom 5
  ]
  node [
    id 1
    label "1"
    cpu 30
    gpu 7
    rom 8
  ]
  node [
    id 2
    label "2"
    cpu 9
    gpu 33
    rom 10
  ]
  node [
    id 3
    label "3"
    cpu 39
    gpu 23
    rom 42
  ]
  node [
    id 4
    label "4"
    cpu 42
    gpu 50
    rom 42
  ]
  node [
    id 5
    label "5"
    cpu 28
    gpu 24
    rom 10
  ]
  node [
    id 6
    label "6"
    cpu 38
    gpu 30
    rom 47
  ]
  node [
    id 7
    label "7"
    cpu 47
    gpu 1
    rom 16
  ]
  node [
    id 8
    label "8"
    cpu 18
    gpu 35
    rom 15
  ]
  node [
    id 9
    label "9"
    cpu 43
    gpu 34
    rom 15
  ]
  edge [
    source 0
    target 1
    bw 28
  ]
  edge [
    source 1
    target 2
    bw 24
  ]
  edge [
    source 2
    target 3
    bw 37
  ]
  edge [
    source 3
    target 4
    bw 44
  ]
  edge [
    source 4
    target 5
    bw 50
  ]
  edge [
    source 5
    target 6
    bw 32
  ]
  edge [
    source 6
    target 7
    bw 19
  ]
  edge [
    source 7
    target 8
    bw 9
  ]
  edge [
    source 8
    target 9
    bw 47
  ]
]
