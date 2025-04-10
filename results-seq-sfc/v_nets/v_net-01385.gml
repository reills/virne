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
  id 1385
  arrival_time 29254.15546719055
  lifetime 542.8881316803369
  num_nodes 8
  type "path"
  node [
    id 0
    label "0"
    cpu 5
    gpu 30
    rom 3
  ]
  node [
    id 1
    label "1"
    cpu 7
    gpu 31
    rom 37
  ]
  node [
    id 2
    label "2"
    cpu 40
    gpu 39
    rom 8
  ]
  node [
    id 3
    label "3"
    cpu 40
    gpu 16
    rom 6
  ]
  node [
    id 4
    label "4"
    cpu 9
    gpu 28
    rom 32
  ]
  node [
    id 5
    label "5"
    cpu 37
    gpu 45
    rom 2
  ]
  node [
    id 6
    label "6"
    cpu 34
    gpu 34
    rom 10
  ]
  node [
    id 7
    label "7"
    cpu 41
    gpu 23
    rom 38
  ]
  edge [
    source 0
    target 1
    bw 21
  ]
  edge [
    source 1
    target 2
    bw 46
  ]
  edge [
    source 2
    target 3
    bw 13
  ]
  edge [
    source 3
    target 4
    bw 50
  ]
  edge [
    source 4
    target 5
    bw 34
  ]
  edge [
    source 5
    target 6
    bw 27
  ]
  edge [
    source 6
    target 7
    bw 21
  ]
]
