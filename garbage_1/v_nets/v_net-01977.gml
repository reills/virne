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
  id 1977
  arrival_time 43175.250609559705
  lifetime 625.5240180040676
  num_nodes 11
  type "path"
  node [
    id 0
    label "0"
    cpu 50
    gpu 5
    rom 0
  ]
  node [
    id 1
    label "1"
    cpu 13
    gpu 16
    rom 39
  ]
  node [
    id 2
    label "2"
    cpu 28
    gpu 5
    rom 14
  ]
  node [
    id 3
    label "3"
    cpu 3
    gpu 23
    rom 11
  ]
  node [
    id 4
    label "4"
    cpu 19
    gpu 47
    rom 38
  ]
  node [
    id 5
    label "5"
    cpu 15
    gpu 24
    rom 13
  ]
  node [
    id 6
    label "6"
    cpu 23
    gpu 21
    rom 29
  ]
  node [
    id 7
    label "7"
    cpu 7
    gpu 27
    rom 24
  ]
  node [
    id 8
    label "8"
    cpu 26
    gpu 47
    rom 27
  ]
  node [
    id 9
    label "9"
    cpu 5
    gpu 13
    rom 26
  ]
  node [
    id 10
    label "10"
    cpu 15
    gpu 35
    rom 20
  ]
  edge [
    source 0
    target 1
    bw 44
  ]
  edge [
    source 1
    target 2
    bw 47
  ]
  edge [
    source 2
    target 3
    bw 40
  ]
  edge [
    source 3
    target 4
    bw 1
  ]
  edge [
    source 4
    target 5
    bw 41
  ]
  edge [
    source 5
    target 6
    bw 9
  ]
  edge [
    source 6
    target 7
    bw 12
  ]
  edge [
    source 7
    target 8
    bw 11
  ]
  edge [
    source 8
    target 9
    bw 4
  ]
  edge [
    source 9
    target 10
    bw 1
  ]
]
