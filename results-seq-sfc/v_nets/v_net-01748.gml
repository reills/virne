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
  id 1748
  arrival_time 38979.31148142558
  lifetime 101.47471258951235
  num_nodes 14
  type "path"
  node [
    id 0
    label "0"
    cpu 19
    gpu 5
    rom 2
  ]
  node [
    id 1
    label "1"
    cpu 49
    gpu 11
    rom 18
  ]
  node [
    id 2
    label "2"
    cpu 23
    gpu 45
    rom 30
  ]
  node [
    id 3
    label "3"
    cpu 14
    gpu 5
    rom 35
  ]
  node [
    id 4
    label "4"
    cpu 0
    gpu 14
    rom 19
  ]
  node [
    id 5
    label "5"
    cpu 38
    gpu 36
    rom 6
  ]
  node [
    id 6
    label "6"
    cpu 44
    gpu 40
    rom 4
  ]
  node [
    id 7
    label "7"
    cpu 44
    gpu 8
    rom 13
  ]
  node [
    id 8
    label "8"
    cpu 20
    gpu 17
    rom 12
  ]
  node [
    id 9
    label "9"
    cpu 49
    gpu 36
    rom 10
  ]
  node [
    id 10
    label "10"
    cpu 39
    gpu 38
    rom 19
  ]
  node [
    id 11
    label "11"
    cpu 47
    gpu 10
    rom 13
  ]
  node [
    id 12
    label "12"
    cpu 36
    gpu 46
    rom 47
  ]
  node [
    id 13
    label "13"
    cpu 29
    gpu 17
    rom 16
  ]
  edge [
    source 0
    target 1
    bw 39
  ]
  edge [
    source 1
    target 2
    bw 41
  ]
  edge [
    source 2
    target 3
    bw 14
  ]
  edge [
    source 3
    target 4
    bw 32
  ]
  edge [
    source 4
    target 5
    bw 2
  ]
  edge [
    source 5
    target 6
    bw 10
  ]
  edge [
    source 6
    target 7
    bw 40
  ]
  edge [
    source 7
    target 8
    bw 11
  ]
  edge [
    source 8
    target 9
    bw 49
  ]
  edge [
    source 9
    target 10
    bw 0
  ]
  edge [
    source 10
    target 11
    bw 33
  ]
  edge [
    source 11
    target 12
    bw 13
  ]
  edge [
    source 12
    target 13
    bw 21
  ]
]
