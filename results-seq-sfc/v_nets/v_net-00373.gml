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
  id 373
  arrival_time 7030.594404392264
  lifetime 678.6139075229786
  num_nodes 10
  type "path"
  node [
    id 0
    label "0"
    cpu 13
    gpu 10
    rom 18
  ]
  node [
    id 1
    label "1"
    cpu 41
    gpu 38
    rom 31
  ]
  node [
    id 2
    label "2"
    cpu 50
    gpu 28
    rom 10
  ]
  node [
    id 3
    label "3"
    cpu 2
    gpu 26
    rom 3
  ]
  node [
    id 4
    label "4"
    cpu 12
    gpu 15
    rom 9
  ]
  node [
    id 5
    label "5"
    cpu 23
    gpu 38
    rom 16
  ]
  node [
    id 6
    label "6"
    cpu 16
    gpu 38
    rom 43
  ]
  node [
    id 7
    label "7"
    cpu 16
    gpu 41
    rom 21
  ]
  node [
    id 8
    label "8"
    cpu 2
    gpu 37
    rom 10
  ]
  node [
    id 9
    label "9"
    cpu 27
    gpu 3
    rom 17
  ]
  edge [
    source 0
    target 1
    bw 46
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
  edge [
    source 3
    target 4
    bw 2
  ]
  edge [
    source 4
    target 5
    bw 17
  ]
  edge [
    source 5
    target 6
    bw 29
  ]
  edge [
    source 6
    target 7
    bw 10
  ]
  edge [
    source 7
    target 8
    bw 15
  ]
  edge [
    source 8
    target 9
    bw 22
  ]
]
