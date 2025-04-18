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
  id 408
  arrival_time 8040.325239926269
  lifetime 148.6196460643138
  num_nodes 12
  type "path"
  node [
    id 0
    label "0"
    cpu 35
    gpu 34
    rom 29
  ]
  node [
    id 1
    label "1"
    cpu 11
    gpu 25
    rom 15
  ]
  node [
    id 2
    label "2"
    cpu 8
    gpu 45
    rom 43
  ]
  node [
    id 3
    label "3"
    cpu 38
    gpu 0
    rom 49
  ]
  node [
    id 4
    label "4"
    cpu 15
    gpu 32
    rom 21
  ]
  node [
    id 5
    label "5"
    cpu 22
    gpu 24
    rom 35
  ]
  node [
    id 6
    label "6"
    cpu 29
    gpu 15
    rom 38
  ]
  node [
    id 7
    label "7"
    cpu 31
    gpu 23
    rom 8
  ]
  node [
    id 8
    label "8"
    cpu 16
    gpu 15
    rom 11
  ]
  node [
    id 9
    label "9"
    cpu 15
    gpu 39
    rom 29
  ]
  node [
    id 10
    label "10"
    cpu 16
    gpu 32
    rom 47
  ]
  node [
    id 11
    label "11"
    cpu 26
    gpu 22
    rom 0
  ]
  edge [
    source 0
    target 1
    bw 9
  ]
  edge [
    source 1
    target 2
    bw 44
  ]
  edge [
    source 2
    target 3
    bw 29
  ]
  edge [
    source 3
    target 4
    bw 21
  ]
  edge [
    source 4
    target 5
    bw 48
  ]
  edge [
    source 5
    target 6
    bw 20
  ]
  edge [
    source 6
    target 7
    bw 14
  ]
  edge [
    source 7
    target 8
    bw 13
  ]
  edge [
    source 8
    target 9
    bw 21
  ]
  edge [
    source 9
    target 10
    bw 24
  ]
  edge [
    source 10
    target 11
    bw 2
  ]
]
