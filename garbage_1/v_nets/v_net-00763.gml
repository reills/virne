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
  id 763
  arrival_time 16049.73027903917
  lifetime 551.5820175334712
  num_nodes 13
  type "path"
  node [
    id 0
    label "0"
    cpu 29
    gpu 27
    rom 16
  ]
  node [
    id 1
    label "1"
    cpu 4
    gpu 48
    rom 2
  ]
  node [
    id 2
    label "2"
    cpu 47
    gpu 0
    rom 17
  ]
  node [
    id 3
    label "3"
    cpu 1
    gpu 1
    rom 44
  ]
  node [
    id 4
    label "4"
    cpu 32
    gpu 22
    rom 4
  ]
  node [
    id 5
    label "5"
    cpu 34
    gpu 45
    rom 14
  ]
  node [
    id 6
    label "6"
    cpu 19
    gpu 20
    rom 7
  ]
  node [
    id 7
    label "7"
    cpu 42
    gpu 34
    rom 14
  ]
  node [
    id 8
    label "8"
    cpu 18
    gpu 41
    rom 29
  ]
  node [
    id 9
    label "9"
    cpu 14
    gpu 19
    rom 14
  ]
  node [
    id 10
    label "10"
    cpu 39
    gpu 32
    rom 26
  ]
  node [
    id 11
    label "11"
    cpu 29
    gpu 2
    rom 37
  ]
  node [
    id 12
    label "12"
    cpu 11
    gpu 24
    rom 31
  ]
  edge [
    source 0
    target 1
    bw 23
  ]
  edge [
    source 1
    target 2
    bw 19
  ]
  edge [
    source 2
    target 3
    bw 7
  ]
  edge [
    source 3
    target 4
    bw 13
  ]
  edge [
    source 4
    target 5
    bw 6
  ]
  edge [
    source 5
    target 6
    bw 19
  ]
  edge [
    source 6
    target 7
    bw 9
  ]
  edge [
    source 7
    target 8
    bw 45
  ]
  edge [
    source 8
    target 9
    bw 36
  ]
  edge [
    source 9
    target 10
    bw 7
  ]
  edge [
    source 10
    target 11
    bw 1
  ]
  edge [
    source 11
    target 12
    bw 33
  ]
]
