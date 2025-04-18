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
  id 192
  arrival_time 3507.1143142057335
  lifetime 1732.3558130704096
  num_nodes 10
  type "path"
  node [
    id 0
    label "0"
    cpu 14
    gpu 31
    rom 24
  ]
  node [
    id 1
    label "1"
    cpu 5
    gpu 31
    rom 39
  ]
  node [
    id 2
    label "2"
    cpu 14
    gpu 16
    rom 43
  ]
  node [
    id 3
    label "3"
    cpu 39
    gpu 6
    rom 48
  ]
  node [
    id 4
    label "4"
    cpu 35
    gpu 17
    rom 15
  ]
  node [
    id 5
    label "5"
    cpu 25
    gpu 34
    rom 17
  ]
  node [
    id 6
    label "6"
    cpu 15
    gpu 44
    rom 27
  ]
  node [
    id 7
    label "7"
    cpu 11
    gpu 25
    rom 20
  ]
  node [
    id 8
    label "8"
    cpu 18
    gpu 35
    rom 18
  ]
  node [
    id 9
    label "9"
    cpu 7
    gpu 50
    rom 10
  ]
  edge [
    source 0
    target 1
    bw 25
  ]
  edge [
    source 1
    target 2
    bw 44
  ]
  edge [
    source 2
    target 3
    bw 8
  ]
  edge [
    source 3
    target 4
    bw 46
  ]
  edge [
    source 4
    target 5
    bw 20
  ]
  edge [
    source 5
    target 6
    bw 3
  ]
  edge [
    source 6
    target 7
    bw 27
  ]
  edge [
    source 7
    target 8
    bw 1
  ]
  edge [
    source 8
    target 9
    bw 43
  ]
]
