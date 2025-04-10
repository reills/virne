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
  id 1520
  arrival_time 33742.43958316451
  lifetime 2603.4034745214717
  num_nodes 10
  type "path"
  node [
    id 0
    label "0"
    cpu 8
    gpu 43
    rom 45
  ]
  node [
    id 1
    label "1"
    cpu 21
    gpu 2
    rom 29
  ]
  node [
    id 2
    label "2"
    cpu 3
    gpu 13
    rom 24
  ]
  node [
    id 3
    label "3"
    cpu 17
    gpu 36
    rom 44
  ]
  node [
    id 4
    label "4"
    cpu 19
    gpu 45
    rom 39
  ]
  node [
    id 5
    label "5"
    cpu 17
    gpu 30
    rom 21
  ]
  node [
    id 6
    label "6"
    cpu 24
    gpu 28
    rom 4
  ]
  node [
    id 7
    label "7"
    cpu 15
    gpu 35
    rom 35
  ]
  node [
    id 8
    label "8"
    cpu 3
    gpu 13
    rom 7
  ]
  node [
    id 9
    label "9"
    cpu 47
    gpu 21
    rom 8
  ]
  edge [
    source 0
    target 1
    bw 9
  ]
  edge [
    source 1
    target 2
    bw 4
  ]
  edge [
    source 2
    target 3
    bw 19
  ]
  edge [
    source 3
    target 4
    bw 21
  ]
  edge [
    source 4
    target 5
    bw 25
  ]
  edge [
    source 5
    target 6
    bw 32
  ]
  edge [
    source 6
    target 7
    bw 0
  ]
  edge [
    source 7
    target 8
    bw 35
  ]
  edge [
    source 8
    target 9
    bw 34
  ]
]
