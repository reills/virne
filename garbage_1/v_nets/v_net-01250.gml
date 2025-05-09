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
  id 1250
  arrival_time 25903.25811372054
  lifetime 367.9873306445038
  num_nodes 9
  type "path"
  node [
    id 0
    label "0"
    cpu 27
    gpu 24
    rom 34
  ]
  node [
    id 1
    label "1"
    cpu 47
    gpu 6
    rom 32
  ]
  node [
    id 2
    label "2"
    cpu 50
    gpu 16
    rom 31
  ]
  node [
    id 3
    label "3"
    cpu 8
    gpu 5
    rom 33
  ]
  node [
    id 4
    label "4"
    cpu 43
    gpu 40
    rom 34
  ]
  node [
    id 5
    label "5"
    cpu 6
    gpu 21
    rom 1
  ]
  node [
    id 6
    label "6"
    cpu 48
    gpu 44
    rom 20
  ]
  node [
    id 7
    label "7"
    cpu 8
    gpu 24
    rom 28
  ]
  node [
    id 8
    label "8"
    cpu 24
    gpu 9
    rom 17
  ]
  edge [
    source 0
    target 1
    bw 7
  ]
  edge [
    source 1
    target 2
    bw 48
  ]
  edge [
    source 2
    target 3
    bw 13
  ]
  edge [
    source 3
    target 4
    bw 31
  ]
  edge [
    source 4
    target 5
    bw 20
  ]
  edge [
    source 5
    target 6
    bw 29
  ]
  edge [
    source 6
    target 7
    bw 42
  ]
  edge [
    source 7
    target 8
    bw 48
  ]
]
