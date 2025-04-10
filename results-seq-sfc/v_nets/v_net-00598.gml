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
  id 598
  arrival_time 12374.162410854844
  lifetime 1265.2866256654945
  num_nodes 10
  type "path"
  node [
    id 0
    label "0"
    cpu 21
    gpu 12
    rom 24
  ]
  node [
    id 1
    label "1"
    cpu 34
    gpu 42
    rom 45
  ]
  node [
    id 2
    label "2"
    cpu 37
    gpu 48
    rom 13
  ]
  node [
    id 3
    label "3"
    cpu 9
    gpu 4
    rom 50
  ]
  node [
    id 4
    label "4"
    cpu 17
    gpu 2
    rom 46
  ]
  node [
    id 5
    label "5"
    cpu 8
    gpu 43
    rom 22
  ]
  node [
    id 6
    label "6"
    cpu 18
    gpu 5
    rom 29
  ]
  node [
    id 7
    label "7"
    cpu 25
    gpu 48
    rom 42
  ]
  node [
    id 8
    label "8"
    cpu 13
    gpu 21
    rom 19
  ]
  node [
    id 9
    label "9"
    cpu 47
    gpu 16
    rom 45
  ]
  edge [
    source 0
    target 1
    bw 40
  ]
  edge [
    source 1
    target 2
    bw 10
  ]
  edge [
    source 2
    target 3
    bw 27
  ]
  edge [
    source 3
    target 4
    bw 9
  ]
  edge [
    source 4
    target 5
    bw 39
  ]
  edge [
    source 5
    target 6
    bw 41
  ]
  edge [
    source 6
    target 7
    bw 30
  ]
  edge [
    source 7
    target 8
    bw 15
  ]
  edge [
    source 8
    target 9
    bw 46
  ]
]
