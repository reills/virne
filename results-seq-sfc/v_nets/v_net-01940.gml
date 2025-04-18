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
  id 1940
  arrival_time 42780.61588802227
  lifetime 330.5833185686259
  num_nodes 9
  type "path"
  node [
    id 0
    label "0"
    cpu 18
    gpu 10
    rom 44
  ]
  node [
    id 1
    label "1"
    cpu 16
    gpu 44
    rom 48
  ]
  node [
    id 2
    label "2"
    cpu 19
    gpu 10
    rom 21
  ]
  node [
    id 3
    label "3"
    cpu 16
    gpu 0
    rom 16
  ]
  node [
    id 4
    label "4"
    cpu 26
    gpu 9
    rom 7
  ]
  node [
    id 5
    label "5"
    cpu 26
    gpu 12
    rom 26
  ]
  node [
    id 6
    label "6"
    cpu 7
    gpu 7
    rom 34
  ]
  node [
    id 7
    label "7"
    cpu 2
    gpu 44
    rom 36
  ]
  node [
    id 8
    label "8"
    cpu 30
    gpu 41
    rom 36
  ]
  edge [
    source 0
    target 1
    bw 2
  ]
  edge [
    source 1
    target 2
    bw 34
  ]
  edge [
    source 2
    target 3
    bw 28
  ]
  edge [
    source 3
    target 4
    bw 40
  ]
  edge [
    source 4
    target 5
    bw 11
  ]
  edge [
    source 5
    target 6
    bw 47
  ]
  edge [
    source 6
    target 7
    bw 35
  ]
  edge [
    source 7
    target 8
    bw 43
  ]
]
