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
  id 546
  arrival_time 10311.393132092806
  lifetime 1671.5117344545192
  num_nodes 13
  type "path"
  node [
    id 0
    label "0"
    cpu 44
    gpu 2
    rom 39
  ]
  node [
    id 1
    label "1"
    cpu 3
    gpu 26
    rom 27
  ]
  node [
    id 2
    label "2"
    cpu 27
    gpu 29
    rom 22
  ]
  node [
    id 3
    label "3"
    cpu 34
    gpu 10
    rom 19
  ]
  node [
    id 4
    label "4"
    cpu 1
    gpu 32
    rom 49
  ]
  node [
    id 5
    label "5"
    cpu 44
    gpu 11
    rom 36
  ]
  node [
    id 6
    label "6"
    cpu 15
    gpu 19
    rom 35
  ]
  node [
    id 7
    label "7"
    cpu 33
    gpu 20
    rom 34
  ]
  node [
    id 8
    label "8"
    cpu 6
    gpu 32
    rom 26
  ]
  node [
    id 9
    label "9"
    cpu 30
    gpu 22
    rom 41
  ]
  node [
    id 10
    label "10"
    cpu 19
    gpu 25
    rom 14
  ]
  node [
    id 11
    label "11"
    cpu 0
    gpu 21
    rom 32
  ]
  node [
    id 12
    label "12"
    cpu 15
    gpu 13
    rom 11
  ]
  edge [
    source 0
    target 1
    bw 38
  ]
  edge [
    source 1
    target 2
    bw 16
  ]
  edge [
    source 2
    target 3
    bw 30
  ]
  edge [
    source 3
    target 4
    bw 30
  ]
  edge [
    source 4
    target 5
    bw 47
  ]
  edge [
    source 5
    target 6
    bw 39
  ]
  edge [
    source 6
    target 7
    bw 34
  ]
  edge [
    source 7
    target 8
    bw 23
  ]
  edge [
    source 8
    target 9
    bw 9
  ]
  edge [
    source 9
    target 10
    bw 3
  ]
  edge [
    source 10
    target 11
    bw 29
  ]
  edge [
    source 11
    target 12
    bw 37
  ]
]
