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
  id 66
  arrival_time 1267.5818614708714
  lifetime 2020.3101867552261
  num_nodes 15
  type "path"
  node [
    id 0
    label "0"
    cpu 5
    gpu 36
    rom 6
  ]
  node [
    id 1
    label "1"
    cpu 0
    gpu 5
    rom 12
  ]
  node [
    id 2
    label "2"
    cpu 3
    gpu 25
    rom 17
  ]
  node [
    id 3
    label "3"
    cpu 37
    gpu 40
    rom 5
  ]
  node [
    id 4
    label "4"
    cpu 1
    gpu 3
    rom 37
  ]
  node [
    id 5
    label "5"
    cpu 26
    gpu 26
    rom 34
  ]
  node [
    id 6
    label "6"
    cpu 24
    gpu 2
    rom 41
  ]
  node [
    id 7
    label "7"
    cpu 44
    gpu 0
    rom 21
  ]
  node [
    id 8
    label "8"
    cpu 50
    gpu 16
    rom 37
  ]
  node [
    id 9
    label "9"
    cpu 5
    gpu 4
    rom 39
  ]
  node [
    id 10
    label "10"
    cpu 9
    gpu 12
    rom 11
  ]
  node [
    id 11
    label "11"
    cpu 28
    gpu 30
    rom 26
  ]
  node [
    id 12
    label "12"
    cpu 3
    gpu 45
    rom 39
  ]
  node [
    id 13
    label "13"
    cpu 14
    gpu 45
    rom 10
  ]
  node [
    id 14
    label "14"
    cpu 28
    gpu 19
    rom 23
  ]
  edge [
    source 0
    target 1
    bw 34
  ]
  edge [
    source 1
    target 2
    bw 15
  ]
  edge [
    source 2
    target 3
    bw 26
  ]
  edge [
    source 3
    target 4
    bw 21
  ]
  edge [
    source 4
    target 5
    bw 33
  ]
  edge [
    source 5
    target 6
    bw 25
  ]
  edge [
    source 6
    target 7
    bw 39
  ]
  edge [
    source 7
    target 8
    bw 35
  ]
  edge [
    source 8
    target 9
    bw 32
  ]
  edge [
    source 9
    target 10
    bw 17
  ]
  edge [
    source 10
    target 11
    bw 34
  ]
  edge [
    source 11
    target 12
    bw 6
  ]
  edge [
    source 12
    target 13
    bw 3
  ]
  edge [
    source 13
    target 14
    bw 31
  ]
]
