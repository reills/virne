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
  id 264
  arrival_time 5212.760171962418
  lifetime 38.05590584708471
  num_nodes 14
  type "path"
  node [
    id 0
    label "0"
    cpu 22
    gpu 50
    rom 37
  ]
  node [
    id 1
    label "1"
    cpu 13
    gpu 4
    rom 8
  ]
  node [
    id 2
    label "2"
    cpu 31
    gpu 7
    rom 9
  ]
  node [
    id 3
    label "3"
    cpu 25
    gpu 6
    rom 47
  ]
  node [
    id 4
    label "4"
    cpu 10
    gpu 16
    rom 18
  ]
  node [
    id 5
    label "5"
    cpu 47
    gpu 38
    rom 30
  ]
  node [
    id 6
    label "6"
    cpu 13
    gpu 22
    rom 28
  ]
  node [
    id 7
    label "7"
    cpu 6
    gpu 4
    rom 38
  ]
  node [
    id 8
    label "8"
    cpu 40
    gpu 21
    rom 1
  ]
  node [
    id 9
    label "9"
    cpu 46
    gpu 15
    rom 37
  ]
  node [
    id 10
    label "10"
    cpu 2
    gpu 30
    rom 11
  ]
  node [
    id 11
    label "11"
    cpu 18
    gpu 23
    rom 39
  ]
  node [
    id 12
    label "12"
    cpu 11
    gpu 26
    rom 41
  ]
  node [
    id 13
    label "13"
    cpu 32
    gpu 19
    rom 39
  ]
  edge [
    source 0
    target 1
    bw 31
  ]
  edge [
    source 1
    target 2
    bw 7
  ]
  edge [
    source 2
    target 3
    bw 15
  ]
  edge [
    source 3
    target 4
    bw 50
  ]
  edge [
    source 4
    target 5
    bw 23
  ]
  edge [
    source 5
    target 6
    bw 20
  ]
  edge [
    source 6
    target 7
    bw 26
  ]
  edge [
    source 7
    target 8
    bw 2
  ]
  edge [
    source 8
    target 9
    bw 6
  ]
  edge [
    source 9
    target 10
    bw 4
  ]
  edge [
    source 10
    target 11
    bw 8
  ]
  edge [
    source 11
    target 12
    bw 47
  ]
  edge [
    source 12
    target 13
    bw 21
  ]
]
