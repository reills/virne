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
  id 1688
  arrival_time 37603.16801017579
  lifetime 213.06559886665744
  num_nodes 11
  type "path"
  node [
    id 0
    label "0"
    cpu 46
    gpu 29
    rom 42
  ]
  node [
    id 1
    label "1"
    cpu 0
    gpu 28
    rom 37
  ]
  node [
    id 2
    label "2"
    cpu 32
    gpu 16
    rom 39
  ]
  node [
    id 3
    label "3"
    cpu 26
    gpu 6
    rom 47
  ]
  node [
    id 4
    label "4"
    cpu 18
    gpu 19
    rom 21
  ]
  node [
    id 5
    label "5"
    cpu 21
    gpu 45
    rom 48
  ]
  node [
    id 6
    label "6"
    cpu 14
    gpu 26
    rom 12
  ]
  node [
    id 7
    label "7"
    cpu 13
    gpu 26
    rom 50
  ]
  node [
    id 8
    label "8"
    cpu 23
    gpu 32
    rom 18
  ]
  node [
    id 9
    label "9"
    cpu 35
    gpu 12
    rom 9
  ]
  node [
    id 10
    label "10"
    cpu 14
    gpu 41
    rom 4
  ]
  edge [
    source 0
    target 1
    bw 28
  ]
  edge [
    source 1
    target 2
    bw 43
  ]
  edge [
    source 2
    target 3
    bw 39
  ]
  edge [
    source 3
    target 4
    bw 6
  ]
  edge [
    source 4
    target 5
    bw 26
  ]
  edge [
    source 5
    target 6
    bw 29
  ]
  edge [
    source 6
    target 7
    bw 47
  ]
  edge [
    source 7
    target 8
    bw 8
  ]
  edge [
    source 8
    target 9
    bw 11
  ]
  edge [
    source 9
    target 10
    bw 2
  ]
]
