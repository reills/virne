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
  id 197
  arrival_time 3546.2220191033507
  lifetime 1359.5725461588634
  num_nodes 15
  type "path"
  node [
    id 0
    label "0"
    cpu 9
    gpu 11
    rom 34
  ]
  node [
    id 1
    label "1"
    cpu 23
    gpu 13
    rom 2
  ]
  node [
    id 2
    label "2"
    cpu 37
    gpu 13
    rom 33
  ]
  node [
    id 3
    label "3"
    cpu 28
    gpu 43
    rom 6
  ]
  node [
    id 4
    label "4"
    cpu 27
    gpu 9
    rom 32
  ]
  node [
    id 5
    label "5"
    cpu 21
    gpu 29
    rom 23
  ]
  node [
    id 6
    label "6"
    cpu 19
    gpu 23
    rom 41
  ]
  node [
    id 7
    label "7"
    cpu 32
    gpu 4
    rom 49
  ]
  node [
    id 8
    label "8"
    cpu 2
    gpu 39
    rom 36
  ]
  node [
    id 9
    label "9"
    cpu 14
    gpu 32
    rom 47
  ]
  node [
    id 10
    label "10"
    cpu 13
    gpu 43
    rom 6
  ]
  node [
    id 11
    label "11"
    cpu 1
    gpu 25
    rom 16
  ]
  node [
    id 12
    label "12"
    cpu 38
    gpu 27
    rom 13
  ]
  node [
    id 13
    label "13"
    cpu 26
    gpu 6
    rom 2
  ]
  node [
    id 14
    label "14"
    cpu 26
    gpu 45
    rom 31
  ]
  edge [
    source 0
    target 1
    bw 15
  ]
  edge [
    source 1
    target 2
    bw 31
  ]
  edge [
    source 2
    target 3
    bw 15
  ]
  edge [
    source 3
    target 4
    bw 19
  ]
  edge [
    source 4
    target 5
    bw 5
  ]
  edge [
    source 5
    target 6
    bw 25
  ]
  edge [
    source 6
    target 7
    bw 27
  ]
  edge [
    source 7
    target 8
    bw 49
  ]
  edge [
    source 8
    target 9
    bw 41
  ]
  edge [
    source 9
    target 10
    bw 16
  ]
  edge [
    source 10
    target 11
    bw 33
  ]
  edge [
    source 11
    target 12
    bw 11
  ]
  edge [
    source 12
    target 13
    bw 44
  ]
  edge [
    source 13
    target 14
    bw 36
  ]
]
