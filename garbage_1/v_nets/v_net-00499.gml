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
  id 499
  arrival_time 9415.757909581393
  lifetime 558.6188620648625
  num_nodes 12
  type "path"
  node [
    id 0
    label "0"
    cpu 30
    gpu 50
    rom 13
  ]
  node [
    id 1
    label "1"
    cpu 32
    gpu 21
    rom 44
  ]
  node [
    id 2
    label "2"
    cpu 45
    gpu 1
    rom 12
  ]
  node [
    id 3
    label "3"
    cpu 8
    gpu 12
    rom 45
  ]
  node [
    id 4
    label "4"
    cpu 7
    gpu 9
    rom 20
  ]
  node [
    id 5
    label "5"
    cpu 3
    gpu 15
    rom 45
  ]
  node [
    id 6
    label "6"
    cpu 7
    gpu 4
    rom 22
  ]
  node [
    id 7
    label "7"
    cpu 10
    gpu 48
    rom 27
  ]
  node [
    id 8
    label "8"
    cpu 50
    gpu 0
    rom 27
  ]
  node [
    id 9
    label "9"
    cpu 13
    gpu 17
    rom 36
  ]
  node [
    id 10
    label "10"
    cpu 29
    gpu 44
    rom 2
  ]
  node [
    id 11
    label "11"
    cpu 24
    gpu 11
    rom 10
  ]
  edge [
    source 0
    target 1
    bw 47
  ]
  edge [
    source 1
    target 2
    bw 9
  ]
  edge [
    source 2
    target 3
    bw 23
  ]
  edge [
    source 3
    target 4
    bw 43
  ]
  edge [
    source 4
    target 5
    bw 37
  ]
  edge [
    source 5
    target 6
    bw 17
  ]
  edge [
    source 6
    target 7
    bw 22
  ]
  edge [
    source 7
    target 8
    bw 10
  ]
  edge [
    source 8
    target 9
    bw 12
  ]
  edge [
    source 9
    target 10
    bw 31
  ]
  edge [
    source 10
    target 11
    bw 22
  ]
]
