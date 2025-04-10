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
  id 1556
  arrival_time 34313.093773986766
  lifetime 954.0992256563501
  num_nodes 14
  type "path"
  node [
    id 0
    label "0"
    cpu 42
    gpu 23
    rom 21
  ]
  node [
    id 1
    label "1"
    cpu 10
    gpu 9
    rom 35
  ]
  node [
    id 2
    label "2"
    cpu 47
    gpu 21
    rom 34
  ]
  node [
    id 3
    label "3"
    cpu 10
    gpu 20
    rom 33
  ]
  node [
    id 4
    label "4"
    cpu 14
    gpu 17
    rom 44
  ]
  node [
    id 5
    label "5"
    cpu 13
    gpu 20
    rom 19
  ]
  node [
    id 6
    label "6"
    cpu 29
    gpu 41
    rom 50
  ]
  node [
    id 7
    label "7"
    cpu 1
    gpu 21
    rom 30
  ]
  node [
    id 8
    label "8"
    cpu 25
    gpu 36
    rom 47
  ]
  node [
    id 9
    label "9"
    cpu 42
    gpu 23
    rom 20
  ]
  node [
    id 10
    label "10"
    cpu 34
    gpu 26
    rom 22
  ]
  node [
    id 11
    label "11"
    cpu 38
    gpu 50
    rom 10
  ]
  node [
    id 12
    label "12"
    cpu 14
    gpu 19
    rom 28
  ]
  node [
    id 13
    label "13"
    cpu 43
    gpu 42
    rom 0
  ]
  edge [
    source 0
    target 1
    bw 23
  ]
  edge [
    source 1
    target 2
    bw 15
  ]
  edge [
    source 2
    target 3
    bw 24
  ]
  edge [
    source 3
    target 4
    bw 0
  ]
  edge [
    source 4
    target 5
    bw 21
  ]
  edge [
    source 5
    target 6
    bw 7
  ]
  edge [
    source 6
    target 7
    bw 15
  ]
  edge [
    source 7
    target 8
    bw 0
  ]
  edge [
    source 8
    target 9
    bw 15
  ]
  edge [
    source 9
    target 10
    bw 3
  ]
  edge [
    source 10
    target 11
    bw 20
  ]
  edge [
    source 11
    target 12
    bw 40
  ]
  edge [
    source 12
    target 13
    bw 8
  ]
]
