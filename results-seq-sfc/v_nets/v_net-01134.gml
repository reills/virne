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
  id 1134
  arrival_time 23615.607359033977
  lifetime 551.2278144365724
  num_nodes 10
  type "path"
  node [
    id 0
    label "0"
    cpu 6
    gpu 48
    rom 12
  ]
  node [
    id 1
    label "1"
    cpu 24
    gpu 46
    rom 40
  ]
  node [
    id 2
    label "2"
    cpu 40
    gpu 21
    rom 36
  ]
  node [
    id 3
    label "3"
    cpu 0
    gpu 37
    rom 0
  ]
  node [
    id 4
    label "4"
    cpu 20
    gpu 24
    rom 3
  ]
  node [
    id 5
    label "5"
    cpu 27
    gpu 6
    rom 33
  ]
  node [
    id 6
    label "6"
    cpu 33
    gpu 32
    rom 29
  ]
  node [
    id 7
    label "7"
    cpu 1
    gpu 15
    rom 18
  ]
  node [
    id 8
    label "8"
    cpu 11
    gpu 47
    rom 15
  ]
  node [
    id 9
    label "9"
    cpu 20
    gpu 37
    rom 50
  ]
  edge [
    source 0
    target 1
    bw 24
  ]
  edge [
    source 1
    target 2
    bw 28
  ]
  edge [
    source 2
    target 3
    bw 12
  ]
  edge [
    source 3
    target 4
    bw 22
  ]
  edge [
    source 4
    target 5
    bw 25
  ]
  edge [
    source 5
    target 6
    bw 33
  ]
  edge [
    source 6
    target 7
    bw 27
  ]
  edge [
    source 7
    target 8
    bw 26
  ]
  edge [
    source 8
    target 9
    bw 1
  ]
]
