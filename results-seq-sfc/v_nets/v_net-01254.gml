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
  id 1254
  arrival_time 25916.85273064844
  lifetime 237.88535900040915
  num_nodes 9
  type "path"
  node [
    id 0
    label "0"
    cpu 5
    gpu 47
    rom 3
  ]
  node [
    id 1
    label "1"
    cpu 7
    gpu 48
    rom 8
  ]
  node [
    id 2
    label "2"
    cpu 45
    gpu 43
    rom 21
  ]
  node [
    id 3
    label "3"
    cpu 46
    gpu 22
    rom 19
  ]
  node [
    id 4
    label "4"
    cpu 34
    gpu 49
    rom 37
  ]
  node [
    id 5
    label "5"
    cpu 42
    gpu 16
    rom 1
  ]
  node [
    id 6
    label "6"
    cpu 1
    gpu 45
    rom 16
  ]
  node [
    id 7
    label "7"
    cpu 38
    gpu 5
    rom 15
  ]
  node [
    id 8
    label "8"
    cpu 50
    gpu 49
    rom 49
  ]
  edge [
    source 0
    target 1
    bw 3
  ]
  edge [
    source 1
    target 2
    bw 15
  ]
  edge [
    source 2
    target 3
    bw 12
  ]
  edge [
    source 3
    target 4
    bw 21
  ]
  edge [
    source 4
    target 5
    bw 12
  ]
  edge [
    source 5
    target 6
    bw 46
  ]
  edge [
    source 6
    target 7
    bw 37
  ]
  edge [
    source 7
    target 8
    bw 38
  ]
]
